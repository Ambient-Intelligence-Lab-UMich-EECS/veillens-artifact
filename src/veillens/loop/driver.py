"""
The protection loop.

`ProtectionLoop` owns the run: it loads the manifest, starts the API workers and the segmentation
thread, drains the batched diffusion queue, and writes the evaluation the paper reports. The four
stages themselves live in `reason/`, `ground/`, `edit/` and -- for verification -- in the gate
inside `loop/api_workers.py`.

Nothing here is specific to geolocation. Which clues are asked for, which of them a mask level
admits, and what "error" means are all supplied by the `PrivacyClassSpec` (see `privacy/`), so the
same loop drives both shipped settings.
"""

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from veillens.edit.pipeline import SDXLInpaintingPipeline
from veillens.loop.api_workers import APIQueueHandler
from veillens.loop.batching import (
    GlobalBatchQueueManager,
)
from veillens.loop.queues import (
    QueueManager,
    create_final_eval_task,
    create_gate_eval_task,
    create_r1_eval_task,
)
from veillens.loop.seg_worker import start_seg_worker
from veillens.loop.state import LEVELS, ClueStatsTracker, GlobalState
from veillens.metrics.geodesic import DistanceCalculator
from veillens.paths import FINAL_DIR, INTERMEDIATE_DIR, RESIZED_INPUTS_DIR, RUN_SUBDIRS
from veillens.privacy.spec import get_spec
from veillens.reason.oracle import PrivacyOracle
from veillens.verify.granularity import target_rung, threshold_km, verdict
from veillens.verify.report import RunVerification
from veillens.vlm.registry import client_from_config

logger = logging.getLogger(__name__)


class ProtectionLoop:
    """
    Main pipeline orchestrator for geolocation clue removal and evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration dictionary, as returned by
                ``veillens.config.load_config`` — placeholders already resolved. Nothing
                downstream re-reads the file, so the dict is the only source of truth.
        """
        self.config = config
        self.output_base = Path(config['output']['base_dir'])
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Create output subdirectories. 'segmentation' and 'final' are per-image and live under
        # intermediate/{image_name}/, so they are created there rather than here.
        self.dirs = {}
        for key, subdir in RUN_SUBDIRS.items():
            dir_path = self.output_base / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            self.dirs[key] = dir_path

        # Phase cache path
        self.phase_cache_path = self.dirs['phase_cache'] / 'phase_cache.json'

        # Resolve privacy class spec (geolocation or personal_info)
        self.spec = get_spec(config)
        print(f"[Pipeline] Privacy class: {self.spec.name}")

        # Initialize components.
        #
        # Two backends: `evaluation:` is the measurement instrument and stays on whatever the
        # config names (GPT-5, historically); `reasoning:` is the attacker inside the loop and may
        # point at a locally served model. A config with no `reasoning:` section gets the
        # evaluation backend for both, which is the pre-split behaviour.
        eval_cfg = config['evaluation']
        reason_cfg = config.get('reasoning') or {}

        eval_client = client_from_config(eval_cfg)
        # trace_dir is injected rather than configured: it belongs to this run, and only the
        # driver knows where that is. A backend with `save_thinking: false` ignores it.
        trace_dir = self.output_base / "reasoning_traces"
        reason_client = (
            client_from_config(reason_cfg, trace_dir=str(trace_dir))
            if reason_cfg else eval_client
        )
        if reason_cfg:
            print(f"[Pipeline] Reasoning backend: {reason_cfg.get('backend', 'openai')} "
                  f"({reason_cfg.get('model', eval_cfg['model'])})")
        print(f"[Pipeline] Evaluation backend: {eval_cfg.get('backend', 'openai')} "
              f"({eval_cfg['model']})")

        self.geo_api = PrivacyOracle(
            model=config['evaluation']['model'],
            # Outer bound across both backends; each client enforces its own limit.
            concurrent_requests=max(
                eval_cfg['concurrent_requests'],
                reason_cfg.get('concurrent_requests', 0),
            ),
            debug_dir=self.output_base / 'debug_cache',
            spec=self.spec,
            reason_client=reason_client,
            eval_client=eval_client,
        )
        self.distance_calc = DistanceCalculator()

        # Model stack: loaded on first use, because --dry-run must not touch the GPU.
        self.sdxl_pipeline = None

        # Dataset and results
        self.dataset = []
        self.phase_cache = {}
        self.evaluation_results = {}
        # Stage IV verdicts, one per image, built as each evaluation entry is created.
        self._verdicts = {}
        # Attacker readings of the *unedited* images, for specs that set `eval_original`. Empty
        # for the settings that take their "before" from the defense oracle's round-1 response.
        self._original_attacker_results = {}

        # Timing tracking
        self.phase_timings = {}

        # Per-clue accounting for end-of-run summary (geolocation spec only).
        self.clue_stats = ClueStatsTracker()

        print(f"[Pipeline] Initialized with output dir: {self.output_base}")

    def _load_phase_cache(self) -> Dict[str, Any]:
        """Load phase cache from disk if it exists."""
        if self.phase_cache_path.exists():
            with open(self.phase_cache_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_phase_cache(self):
        """Save phase cache to disk."""
        with open(self.phase_cache_path, 'w') as f:
            json.dump(self.phase_cache, f, indent=2)
        print(f"[Pipeline] Saved phase cache to {self.phase_cache_path}")

    def _save_evaluation_results(self):
        """
        Save evaluation results to disk.

        This runs after every image is finished, so a raised exception here throws away
        the measurement for an entire run's worth of GPU time. `default` coerces anything
        json cannot encode -- in practice a stray numpy scalar -- instead of failing the
        job over one value. Producers should still hand over plain Python types.
        """
        def _plain(obj):
            item = getattr(obj, "item", None)  # numpy scalars -> int/float/bool
            if callable(item):
                return item()
            logger.warning(f"[Pipeline] Coerced un-encodable {type(obj).__name__} to str")
            return str(obj)

        results_path = self.dirs['evaluation_results'] / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=_plain)
        print(f"[Pipeline] Saved evaluation results to {results_path}")

    # ========================================================================
    # PHASE 1: Dataset Loading
    # ========================================================================

    def load_dataset(self):
        """
        Phase 1: Load dataset from JSON file.
        """
        print("\n" + "=" * 60)
        print("PHASE 1: Dataset Loading")
        print("=" * 60)

        from veillens.datasets.schema import load_manifest

        dataset_path = Path(self.config['dataset']['dataset_json_file'])
        # Handles both schema versions and resolves every image_path against $VEILLENS_DATA.
        self.dataset = load_manifest(dataset_path)

        print(f"[Phase 1] Loaded {len(self.dataset)} images from {dataset_path}")

        # Initialize phase cache for each image
        self.phase_cache = self._load_phase_cache()
        for item in self.dataset:
            image_id = item['image_id']
            if image_id not in self.phase_cache:
                self.phase_cache[image_id] = {
                    'prediction': None,
                    'masks': [],
                    'final': None,
                    'evaluation': None
                }

        self._save_phase_cache()

        # Cache original images in intermediate folders for reference
        print("[Phase 1] Caching original images...")
        cached_count = 0
        cleared_count = 0
        resized_count = 0
        for item in self.dataset:
            image_id = item['image_id']
            image_path = Path(item['image_path'])
            image_name = image_path.stem

            # A run overwrites the files it produces but cannot overwrite files it never
            # produces. An earlier, longer run of the same image leaves its extra
            # iter{N} snapshots and segmentation/iter{N}_* masks behind, and the next
            # reader -- the scorer, the demo exporter -- has no way to tell them from
            # this run's output and will happily mix two runs together. Clearing here
            # makes "always start fresh" (see _initialize_all_states_parallel) true.
            intermediate_dir = self.output_base / INTERMEDIATE_DIR / image_name
            if intermediate_dir.exists():
                shutil.rmtree(intermediate_dir)
                cleared_count += 1
            intermediate_dir.mkdir(parents=True, exist_ok=True)

            # A privacy class may cap input resolution. When it does, the run processes the
            # downscaled copy and `image_path` is repointed at it here -- which is what puts
            # every later reader (segmentation, depth, SD, the attacker) in one coordinate
            # space, and what makes `original.*` below the image that was actually processed.
            source_path = self._stage_input(image_path)
            if source_path != image_path:
                item['image_path'] = str(source_path)
                resized_count += 1

            original_cache_path = intermediate_dir / f"original{source_path.suffix}"
            shutil.copy2(source_path, original_cache_path)
            cached_count += 1

        if cleared_count:
            print(f"[Phase 1] Cleared {cleared_count} stale intermediate folders")
        print(f"[Phase 1] Cached {cached_count} original images")
        if resized_count:
            print(f"[Phase 1] Scaled {resized_count} input(s) to a "
                  f"{self.spec.max_input_dim}px long side (see {RESIZED_INPUTS_DIR}/)")

    def _stage_input(self, image_path: Path) -> Path:
        """
        The image this run will actually process.

        Returns *image_path* unchanged unless the privacy class caps input resolution
        (:attr:`PrivacyClassSpec.max_input_dim`), in which case a copy scaled so its long side
        equals the cap is written under ``resized_inputs/`` and that path is returned instead.

        The cap is not a performance tweak. Nothing in the edit path bounds resolution --
        :class:`SDXLInpaintingPipeline` inpaints at whatever size it is handed -- so a
        multi-megapixel source asks for a single allocation large enough to exhaust a 48 GB GPU.
        Scaling once here, rather than at inpaint time, keeps the masks, the depth map and the
        attacker's reading in one coordinate space.

        The staged file keeps the image's own filename: ``eval_cache`` entries are named from
        this path's stem, so staging under a shared name would collide across images.
        """
        cap = self.spec.max_input_dim
        if not cap:
            return image_path

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= cap:
            return image_path

        scale = cap / longest
        resized = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,       # the correct filter for downscaling
        )

        staged_dir = self.output_base / RESIZED_INPUTS_DIR
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staged_dir / image_path.name
        params = ([cv2.IMWRITE_JPEG_QUALITY, 95]
                  if staged_path.suffix.lower() in {".jpg", ".jpeg"} else [])
        if not cv2.imwrite(str(staged_path), resized, params):
            raise RuntimeError(f"Could not write staged input: {staged_path}")
        return staged_path

    # ========================================================================
    # PHASE 2: Original Evaluation
    # ========================================================================

    async def run_parallel_pipeline(self):
        """
        Run the pipeline with complete parallelism between API and GPU.

        This method implements the new queue-based architecture:
        - api_queue: Handles all GPT API requests (async workers)
        - seg_queue: Handles GPU segmentation (single worker thread)
        - sd_queue: Handles batched SDXL inpainting

        Workflow:
        1. Initialize all queues and global state
        2. Consolidate all cache checks in initialization
        3. Start API workers (async) and segmentation worker (thread)
        4. Monitor SD queue and flush when seg_queue empty for timeout
        5. Handle compositing and final eval triggers
        6. Wait for all images to complete
        """
        print("\n" + "=" * 60)
        print("PARALLEL PIPELINE: Complete Parallelism Mode")
        print("=" * 60)

        # Load Grounding DINO, SAM 2, MiDaS and SDXL.
        if self.sdxl_pipeline is None:
            self.sdxl_pipeline = SDXLInpaintingPipeline(self.config)

        # Initialize global state and queue manager
        self.global_state = GlobalState()
        self.queue_manager = QueueManager(self.config)

        # Initialize SD queue (reuse GlobalBatchQueueManager)
        sd_queue = GlobalBatchQueueManager(
            round=1,  # Will handle both rounds
            config=self.config,
            flush_strategy="hybrid"
        )
        self.queue_manager.set_sd_queue(sd_queue)

        # Step 1: Initialize all image states and dispatch initial tasks
        self._initialize_all_states_parallel()

        total_images = self.global_state.total_images
        completed, _ = self.global_state.get_progress()

        logger.info(f"[Parallel] Total images: {total_images}")
        logger.info(f"[Parallel] Already complete: {completed}")
        logger.info(f"[Parallel] To process: {total_images - completed}")

        if self.global_state.is_all_complete():
            logger.info("[Parallel] All images already complete, nothing to do")
            return

        # Step 2: Start API workers (async)
        event_loop = asyncio.get_event_loop()
        # Size the pool from `reasoning:`, not `evaluation:`. Every worker spends almost all of
        # its time on R1/R2/gate calls, which go to the reasoning backend; the final eval is one
        # call per image. Reading `evaluation.concurrent_requests` here meant the local server was
        # throttled to a vendor rate limit it is not subject to, and made
        # `reasoning.concurrent_requests` silently dead config.
        reason_cfg = self.config.get('reasoning') or {}
        num_api_workers = reason_cfg.get(
            'concurrent_requests',
            self.config['evaluation'].get('concurrent_requests', 3)
        )

        api_handler = APIQueueHandler(
            self.queue_manager,
            self.global_state,
            self.geo_api,
            self.config
        )
        api_workers = [
            asyncio.create_task(api_handler.run(worker_id=i))
            for i in range(num_api_workers)
        ]
        logger.info(f"[Parallel] Started {num_api_workers} API workers")

        # Step 3: Start segmentation worker (thread)
        # Daemon thread; the run ends when the driver's own loop finishes, so no join.
        start_seg_worker(
            self.queue_manager,
            self.global_state,
            self.sdxl_pipeline,
            self.config,
            event_loop,
            self.clue_stats
        )
        logger.info("[Parallel] Started segmentation worker thread")

        # Step 4: Main loop - monitor SD queue and process batches
        try:
            await self._parallel_main_loop()
        finally:
            # Cleanup
            self.queue_manager.shutdown()
            for worker in api_workers:
                worker.cancel()
            logger.info("[Parallel] Cleanup complete")

        # Save final phase cache
        self._sync_global_state_to_phase_cache()
        self._save_phase_cache()

        completed, total = self.global_state.get_progress()
        logger.info(f"[Parallel] Pipeline complete: {completed}/{total} images")

    def _initialize_all_states_parallel(self):
        """
        Initialize all image states and dispatch initial tasks.

        Simplified version: No cache reading, always start fresh.
        For each image:
        1. Create state
        2. Push R1 eval task

        All existing files will be overwritten during processing.
        """
        logger.info("[Init] Initializing fresh pipeline (no cache reading)...")

        for item in self.dataset:
            image_id = item['image_id']
            image_path = item['image_path']

            # Create state
            self.global_state.add_image(
                image_id=image_id,
                image_path=image_path,
                ground_truth=item.get(self.spec.ground_truth_field),
                item=item,
            )

            # Push R1 eval task for every image
            task = create_r1_eval_task(image_id, image_path)
            self.queue_manager.put_api_task_nowait(task)
            logger.debug(f"[Init] {image_id}: Pushed R1 eval task")

        # Set total images
        self.global_state.set_total_images(len(self.dataset))
        logger.info(f"[Init] Initialized {len(self.dataset)} image states, all starting fresh")

    async def _parallel_main_loop(self):
        """
        Main event loop for parallel pipeline.

        Monitors queues and processes SD batches when appropriate.
        """
        check_interval = 1.0  # seconds
        log_interval = 10.0  # seconds
        last_log_time = time.time()

        # This loop's only exit is every image being counted complete, so any image that leaves
        # the state machine without being counted hangs the job until walltime -- eight hours of
        # a held GPU producing nothing. Two such paths have been found and fixed; rather than
        # trust that there is no third, detect the condition itself.
        #
        # The signal is a fingerprint of everything that moves. Slowness does not trip it: a
        # single queue transition or one completed image resets the clock. Only genuine deadlock,
        # where nothing changes at all, runs it out. The default is long enough to cover a cold
        # model server (measured: ~700 s of no activity at all while vLLM loads).
        stall_timeout = float(self.config.get('experiment', {}).get('stall_timeout', 1800))
        last_fingerprint = None
        last_change_time = time.time()

        while not self.global_state.is_all_complete():
            current_time = time.time()

            # SDXL runs only after segmentation has been idle long enough.
            seg_empty_duration = self.queue_manager.get_seg_empty_duration()

            ready_batches = self.queue_manager.sd_queue.get_ready_batches(
                seg_empty_duration=seg_empty_duration,
                timeout=self.queue_manager.sd_flush_timeout
            )

            if ready_batches:
                self.queue_manager.reset_seg_empty_timer()
                for size_class, batch_items in ready_batches:
                    await self._process_sd_batch(size_class, batch_items)

            completed, total = self.global_state.get_progress()
            status = self.queue_manager.get_status()

            fingerprint = (
                completed,
                status['api_queue_size'],
                status['seg_queue_size'],
                status.get('sd_queue_total', 0),
                bool(ready_batches),
            )
            if fingerprint != last_fingerprint:
                last_fingerprint = fingerprint
                last_change_time = current_time
            elif current_time - last_change_time > stall_timeout:
                stuck = self.global_state.get_pending_images()
                logger.error(
                    f"[Stall] No progress for {stall_timeout:.0f}s at {completed}/{total} "
                    f"complete with all queues idle. The remaining images left the state machine "
                    f"without being counted, so this loop would never exit. Stopping so the run "
                    f"writes what it has. Unfinished: {stuck}"
                )
                break

            # Periodic logging
            if current_time - last_log_time >= log_interval:
                logger.info(
                    f"[Progress] {completed}/{total} complete | "
                    f"API: {status['api_queue_size']} | "
                    f"Seg: {status['seg_queue_size']} | "
                    f"SD: {status.get('sd_queue_total', 0)}"
                )
                last_log_time = current_time

            await asyncio.sleep(check_interval)

        # Final flush of any remaining SD items
        remaining_batches = self.queue_manager.sd_queue.get_all_batches()
        for size_class, batch_items in remaining_batches:
            if batch_items:
                await self._process_sd_batch(size_class, batch_items)

    async def _process_sd_batch(self, size_class: str, batch_items):
        """Process a batch of items through SDXL and composite results."""
        logger.info(f"[SD Batch] Processing {len(batch_items)} {size_class} items")

        sd_items = batch_items
        if not sd_items:
            return

        # Set SD running flag to pause seg worker (GPU coordination)
        self.queue_manager.set_sd_running(True)
        logger.debug("[SD Batch] Set GPU lock - seg worker will pause")

        try:
            # Prepare batch inputs
            images = [item.image_crop for item in sd_items]
            masks = [item.mask_crop for item in sd_items]
            depth_maps = [item.depth_crop for item in sd_items]
            prompts = [item.positive_prompt for item in sd_items]
            neg_prompts = [item.negative_prompt for item in sd_items]

            # Run SDXL inpainting in thread pool to avoid blocking event loop
            batch_results = await asyncio.to_thread(
                self.sdxl_pipeline.inpainter.inpaint_batch,
                images, masks, depth_maps, prompts, neg_prompts
            )

            # Composite results and check completion
            for item, result in zip(sd_items, batch_results):
                await self._composite_and_check_completion(item, result)

        finally:
            # Clear SD running flag (allow seg worker to resume)
            self.queue_manager.set_sd_running(False)
            logger.debug("[SD Batch] Released GPU lock - seg worker can resume")

    async def _composite_and_check_completion(self, item, inpainted_result):
        """Composite single result and check if image is complete."""
        state = self.global_state.get_image(item.image_id)
        if not state:
            return

        image_name = state.image_name

        # Load or create intermediate image
        intermediate_dir = self.output_base / INTERMEDIATE_DIR / image_name / "working"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = intermediate_dir / f"{image_name}_working.png"

        if intermediate_path.exists():
            working_image = await asyncio.to_thread(cv2.imread, str(intermediate_path))
            working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)
        else:
            # Start from original image
            working_image = await asyncio.to_thread(cv2.imread, state.image_path)
            working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)

        # Composite the inpainted result
        working_image = self._composite_single_result(
            working_image, inpainted_result, item
        )

        # Save working image
        working_bgr = cv2.cvtColor(working_image, cv2.COLOR_RGB2BGR)
        await asyncio.to_thread(cv2.imwrite, str(intermediate_path), working_bgr)

        # Track mask path
        state.add_mask_path(item.mask_path)

        # Increment finished count
        state.increment_finished_clue()

        # Check iter completion
        if state.is_complete():
            iter_idx = state.r2_iter
            final_dir = self.output_base / INTERMEDIATE_DIR / image_name / FINAL_DIR
            final_dir.mkdir(parents=True, exist_ok=True)

            # Per-iter snapshot of the current working image
            iter_snapshot = final_dir / f"iter{iter_idx}_{image_name}_final.png"
            await asyncio.to_thread(cv2.imwrite, str(iter_snapshot), working_bgr)

            max_iter = int(self.config.get('experiment', {}).get('max_iter', 1))

            if (not state.stop_after_current_iter) and iter_idx < max_iter:
                # Advance: reset per-iter R2 state, bump iter, push next gate eval
                state.reset_for_next_iter()
                state.r2_iter = iter_idx + 1
                gate_task = create_gate_eval_task(
                    item.image_id, str(intermediate_path), iter_idx=iter_idx + 1
                )
                await self.queue_manager.put_api_task(gate_task)
                logger.info(
                    f"[Iter {iter_idx} done] {item.image_id}: pushed gate_eval(iter={iter_idx + 1})"
                )
            else:
                # Finalize: write canonical final.png and push final_eval
                final_path = final_dir / f"{image_name}_final.png"
                await asyncio.to_thread(cv2.imwrite, str(final_path), working_bgr)
                state.final_image = str(final_path)

                task = create_final_eval_task(item.image_id, str(final_path))
                await self.queue_manager.put_api_task(task)
                logger.info(
                    f"[Complete] {item.image_id}: finalized after iter {iter_idx}, pushed final_eval"
                )

    def _composite_single_result(self, working_image, inpainted_crop, item):
        """Composite single inpainted crop onto working image."""
        # Extract position from item
        x1, y1 = item.crop_x1, item.crop_y1
        crop_w, crop_h = item.crop_w, item.crop_h
        pad_top, pad_bottom = item.pad_top, item.pad_bottom
        pad_left, pad_right = item.pad_left, item.pad_right

        # Remove padding from inpainted result
        if pad_top or pad_bottom or pad_left or pad_right:
            h, w = inpainted_crop.shape[:2]
            inpainted_crop = inpainted_crop[
                pad_top:h - pad_bottom if pad_bottom else h,
                pad_left:w - pad_right if pad_right else w
            ]

        # Get mask region
        mask = item.original_mask
        y2, x2 = y1 + crop_h, x1 + crop_w

        # Ensure bounds
        h, w = working_image.shape[:2]
        y2 = min(y2, h)
        x2 = min(x2, w)

        # Extract mask region for compositing
        mask_region = mask[y1:y2, x1:x2]

        # Resize inpainted to match region if needed
        region_h, region_w = y2 - y1, x2 - x1
        if inpainted_crop.shape[:2] != (region_h, region_w):
            inpainted_crop = cv2.resize(
                inpainted_crop, (region_w, region_h),
                interpolation=cv2.INTER_LANCZOS4
            )

        # Alpha blend
        mask_norm = mask_region.astype(np.float32) / 255.0
        if len(mask_norm.shape) == 2:
            mask_norm = mask_norm[:, :, np.newaxis]

        working_region = working_image[y1:y2, x1:x2].astype(np.float32)
        inpainted_region = inpainted_crop.astype(np.float32)

        blended = working_region * (1 - mask_norm) + inpainted_region * mask_norm
        working_image[y1:y2, x1:x2] = blended.astype(np.uint8)

        return working_image

    def _sync_global_state_to_phase_cache(self):
        """Sync global state back to phase_cache for persistence."""
        for image_id, state in self.global_state.images.items():
            if image_id not in self.phase_cache:
                self.phase_cache[image_id] = {
                    'prediction': None,
                    'masks': [],
                    'final': None,
                    'evaluation': None
                }

            self.phase_cache[image_id]['prediction'] = state.r1_pred_path
            self.phase_cache[image_id]['masks'] = state.mask_paths
            self.phase_cache[image_id]['final'] = state.final_image
            self.phase_cache[image_id]['evaluation'] = state.final_eval

    # ========================================================================
    # PHASE 5: Final Evaluation and Reporting
    # ========================================================================

    async def evaluate_processed_images(self):
        """
        Phase 5: Evaluate processed images and generate comparison report.
        """
        print("\n" + "=" * 60)
        print("PHASE 5: Processed Image Evaluation and Reporting")
        print("=" * 60)

        # Collect images to evaluate
        to_evaluate = []
        for item in self.dataset:
            image_id = item['image_id']
            final_path = self.phase_cache[image_id]['final']

            if final_path is None:
                print(f"[Phase 5] No processed image for {image_id}, skipping")
                continue

            to_evaluate.append((item, final_path))

        if not to_evaluate:
            print("[Phase 5] No processed images to evaluate")
            return

        print(f"[Phase 5] Evaluating {len(to_evaluate)} processed images...")

        # Which schema the attacker answers under is the spec's to say: the compositional
        # setting reuses its round-1 schema, which already exposes the privacy score we need.
        tasks = []
        for item, final_path in to_evaluate:
            image_id = item['image_id']
            task = self.geo_api.get_geolocation_prediction(
                final_path,
                schema_type=self.spec.eval_schema_type,
                role='eval',
                prompt_context=self._item_context(item),
            )
            tasks.append((image_id, item, task))

        # Execute all tasks concurrently using gather
        results = await asyncio.gather(*[task for _, _, task in tasks])

        # Some settings need the "before" reading taken by the attacker rather than by the
        # defense oracle -- see _read_originals_with_attacker.
        await self._read_originals_with_attacker(to_evaluate)

        # Process results
        for (image_id, item, _), result in zip(tasks, results):
            if result:
                # Save eval_cache (restore removed caching functionality)
                image_name = Path(item['image_path']).stem
                eval_cache_file = self.dirs['eval_cache'] / f"{image_name}_eval.json"
                with open(eval_cache_file, 'w') as f:
                    json.dump(result, f, indent=2)

                # Calculate errors and create evaluation entry
                self._create_evaluation_entry(item, result)
                print(f"[Phase 5] Evaluated {image_id}, saved to {eval_cache_file}")
            else:
                print(f"[Phase 5] Failed to evaluate {image_id}")

        self._save_evaluation_results()
        self._save_verification()
        self._print_summary_statistics()

        print("[Phase 5] Completed evaluation and reporting")

    def _item_context(self, item: Dict) -> Optional[str]:
        """The per-image prompt suffix for this setting, or None if it needs none."""
        if self.spec.format_item_context is None:
            return None
        return self.spec.format_item_context(item)

    async def _read_originals_with_attacker(self, to_evaluate):
        """
        Measure the unedited image with the *attacker*, for settings whose headline is before/after.

        By default the "before" reading is the defense oracle's round-1 response, which is free --
        the loop already paid for it. That is fine when the reported quantity is an error the two
        models measure the same way, and wrong when the reported quantity is one adversary's
        accuracy before versus after: attributing the raw number to the defense oracle compares
        two different models and calls the difference protection.

        So a spec can ask for one extra attacker call per image. Results land in
        ``eval_cache/<name>_orig_eval.json`` and are picked up by `_create_evaluation_entry`.
        """
        if not self.spec.eval_original:
            return

        print(f"[Phase 5] Reading {len(to_evaluate)} unedited images with the attacker "
              "(spec.eval_original)")
        tasks = [
            self.geo_api.get_geolocation_prediction(
                item['image_path'],
                schema_type=self.spec.eval_schema_type,
                role='eval',
                prompt_context=self._item_context(item),
            )
            for item, _ in to_evaluate
        ]
        results = await asyncio.gather(*tasks)

        for (item, _), result in zip(to_evaluate, results):
            if not result:
                # Left absent rather than faked. `_create_evaluation_entry` falls back to the
                # oracle's round-1 reading and records `read_by` so the results file says which
                # model produced this image's "before" number; writing nothing here would drop
                # the image from the comparison instead.
                print(f"[Phase 5] Attacker failed on the unedited {item['image_id']}")
                continue
            path = self.dirs['eval_cache'] / f"{Path(item['image_path']).stem}_orig_eval.json"
            with open(path, 'w') as handle:
                json.dump(result, handle, indent=2)
            self._original_attacker_results[item['image_id']] = result

    def _create_evaluation_entry(self, item: Dict, processed_result: Dict):
        """
        Create evaluation entry comparing original and processed predictions.

        Field names (`error_km`/`error_score`, `ground_truth_coords`/`ground_truth_score`)
        and the error metric itself (haversine vs. absolute difference) are determined
        by the active PrivacyClassSpec.

        Args:
            item: Dataset item
            processed_result: GPT prediction for processed image
        """
        image_id = item['image_id']
        ground_truth = self.spec.normalize_ground_truth(item[self.spec.ground_truth_field])

        # The "before" reading. When the spec asked for it, this is the attacker on the
        # unedited image; otherwise it is the defense oracle's round-1 response, which the loop
        # produced anyway. See _read_originals_with_attacker.
        original_result = self._original_attacker_results.get(image_id)
        before_read_by = "attacker"
        if original_result is None:
            before_read_by = "defense oracle"
            original_pred_path = self.phase_cache[image_id]['prediction']
            with open(original_pred_path, 'r') as f:
                original_result = json.load(f)

        # Calculate errors via the spec
        original_error = self.spec.compute_error(ground_truth, original_result)
        processed_error = self.spec.compute_error(ground_truth, processed_result)
        error_increase = processed_error - original_error

        # Count mask pixels using combined mask to avoid double-counting overlaps
        mask_paths = self.phase_cache[image_id]['masks']
        combined_mask = None
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if combined_mask is None:
                    combined_mask = (mask > 0).astype(np.uint8)
                else:
                    # Use logical OR to combine masks (max of binary values)
                    combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8))

        # int()/float() rather than the numpy scalars: these go straight into
        # evaluation_results.json, and json.dump refuses np.int64. numpy 2 returns one from
        # count_nonzero where numpy 1 returned a Python int, which is enough to lose a whole
        # run's results at the last step -- after all the GPU work is already paid for.
        total_mask_pixels = int(np.count_nonzero(combined_mask)) if combined_mask is not None else 0

        # Calculate mask percentage
        image_path = item['image_path']
        img = cv2.imread(image_path)
        total_pixels = img.shape[0] * img.shape[1]
        mask_percentage = float(total_mask_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

        # Stage IV: does the finished image actually meet the granularity the run targeted?
        # Both criteria are recorded per image -- the distance one needs the dataset label and so
        # is an offline measurement; the residual one reads the granularity labels the attacker
        # itself emitted and needs no label at all.
        image_verdict = verdict(
            image_id,
            privacy_class=self.spec.name,
            mask_level=self.config['experiment']['mask_level'],
            error=processed_error,
            attacker_result=processed_result,
        )
        self._verdicts[image_id] = image_verdict

        # Create evaluation entry
        self.evaluation_results[image_id] = {
            self.spec.gt_output_key: self.spec.serialize_ground_truth(ground_truth),
            'feature_results': [
                {
                    'feature_removed': 'none',
                    # Which model produced this reading. Normally the defense oracle's round-1
                    # response, which the loop paid for anyway; 'attacker' when the spec asked
                    # for a separate adversary pass, or a mix if that pass failed on some images.
                    # A before/after claim is only as good as the two readings coming from the
                    # same model, so the file records it rather than leaving it to be assumed.
                    'read_by': before_read_by,
                    self.spec.error_key: original_error,
                    'gpt_response': self.spec.extract_prediction_summary(original_result),
                    self.spec.error_increase_key: 0.0,
                    'mask_pixels': 0,
                    'mask_percentage': 0.0
                },
                {
                    'feature_removed': self.spec.format_mask_level_label(
                        self.config['experiment']['mask_level']
                    ),
                    self.spec.error_key: processed_error,
                    'gpt_response': self.spec.extract_prediction_summary(processed_result),
                    self.spec.error_increase_key: error_increase,
                    'mask_pixels': total_mask_pixels,
                    'mask_percentage': mask_percentage,
                    'verification': image_verdict.as_json()
                }
            ]
        }

    def _print_summary_statistics(self):
        """Print summary statistics of the evaluation."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        original_errors = []
        processed_errors = []
        error_increases = []

        err_key = self.spec.error_key
        inc_key = self.spec.error_increase_key
        unit = self.spec.error_unit

        for image_id, result in self.evaluation_results.items():
            features = result['feature_results']
            original_errors.append(features[0][err_key])
            processed_errors.append(features[1][err_key])
            error_increases.append(features[1][inc_key])

        # Calculate statistics
        original_stats = self.distance_calc.calculate_error_statistics(original_errors)
        processed_stats = self.distance_calc.calculate_error_statistics(processed_errors)

        print("\nOriginal Images:")
        print(f"  Mean Error: {original_stats['mean']:.4f} {unit}")
        print(f"  Median Error: {original_stats['median']:.4f} {unit}")
        print(f"  Std Dev: {original_stats['std_dev']:.4f} {unit}")

        print("\nProcessed Images:")
        print(f"  Mean Error: {processed_stats['mean']:.4f} {unit}")
        print(f"  Median Error: {processed_stats['median']:.4f} {unit}")
        print(f"  Std Dev: {processed_stats['std_dev']:.4f} {unit}")

        print("\nError Increase:")
        print(f"  Mean: {np.mean(error_increases):.4f} {unit}")
        print(f"  Median: {np.median(error_increases):.4f} {unit}")
        print(f"  Max: {np.max(error_increases):.4f} {unit}")
        print(f"  Min: {np.min(error_increases):.4f} {unit}")

        # Calculate improvement/degradation
        improved = sum(1 for e in error_increases if e < 0)
        degraded = sum(1 for e in error_increases if e > 0)
        unchanged = len(error_increases) - improved - degraded

        print("\nPerformance:")
        print(f"  Improved: {improved} ({improved / len(error_increases) * 100:.1f}%)")
        print(f"  Degraded: {degraded} ({degraded / len(error_increases) * 100:.1f}%)")
        print(f"  Unchanged: {unchanged} ({unchanged / len(error_increases) * 100:.1f}%)")

    def _print_clue_stats(self):
        """Print per-mask-level clue accounting (geolocation spec only).

        Each clue is counted once at its most specific level, so per-row
        sums equal the distinct clue count and grounded/selected gives
        the grounding success rate.
        """
        if self.spec.name != "geolocation":
            return
        if not self.clue_stats.has_data():
            return

        selected, skipped = self.clue_stats.snapshot()
        mask_level = self.config['experiment'].get('mask_level', [])

        short = {
            "country level clue": "country",
            "city level clue": "city",
            "street level clue": "street",
        }

        print("\n" + "=" * 60)
        print(f"CLUE STATISTICS (geolocation, mask_level={mask_level})")
        print("=" * 60)

        col_w = 10
        header = f"{'':28s}" + "".join(f"{short[l]:>{col_w}s}" for l in LEVELS) + f"{'total':>{col_w}s}" + f"{'rate':>{col_w}s}"
        print(header)

        def fmt_row(label: str, per_level: Dict[str, int], rate: Optional[float]) -> str:
            total = sum(per_level.values())
            cells = "".join(f"{per_level[l]:>{col_w}d}" for l in LEVELS)
            rate_str = f"{rate*100:>{col_w-1}.1f}%" if rate is not None else f"{'-':>{col_w}s}"
            return f"{label:28s}{cells}{total:>{col_w}d}{rate_str}"

        def per_level(bucket: Dict[Tuple[str, str], int], round_tag: str) -> Dict[str, int]:
            return {l: bucket[(round_tag, l)] for l in LEVELS}

        for round_tag, label_round in (("R1", "Round 1"), ("R2", "Round 2")):
            sel = per_level(selected, round_tag)
            skp = per_level(skipped, round_tag)
            grounded = {l: sel[l] - skp[l] for l in LEVELS}
            sel_total = sum(sel.values())
            grd_total = sum(grounded.values())
            rate = (grd_total / sel_total) if sel_total > 0 else None
            print(fmt_row(f"{label_round} selected:", sel, None))
            print(fmt_row(f"{label_round} skipped (no ground):", skp, None))
            print(fmt_row(f"{label_round} grounded:", grounded, rate))

        print("-" * 60)
        totals_sel = {l: selected[("R1", l)] + selected[("R2", l)] for l in LEVELS}
        totals_skp = {l: skipped[("R1", l)] + skipped[("R2", l)] for l in LEVELS}
        totals_grd = {l: totals_sel[l] - totals_skp[l] for l in LEVELS}
        sel_sum = sum(totals_sel.values())
        grd_sum = sum(totals_grd.values())
        total_rate = (grd_sum / sel_sum) if sel_sum > 0 else None
        print(fmt_row("Total selected:", totals_sel, None))
        print(fmt_row("Total skipped (no ground):", totals_skp, None))
        print(fmt_row("Total grounded:", totals_grd, total_rate))
        print("=" * 60)

    def _save_clue_stats(self):
        """Persist clue accounting (distinct counts + grounding rate) to result folder."""
        if self.spec.name != "geolocation":
            return
        if not self.clue_stats.has_data():
            return

        selected, skipped = self.clue_stats.snapshot()
        mask_level = self.config['experiment'].get('mask_level', [])

        def per_level(bucket: Dict[Tuple[str, str], int], round_tag: str) -> Dict[str, int]:
            return {l: bucket[(round_tag, l)] for l in LEVELS}

        def round_block(round_tag: str) -> Dict:
            sel = per_level(selected, round_tag)
            skp = per_level(skipped, round_tag)
            grd = {l: sel[l] - skp[l] for l in LEVELS}
            sel_total = sum(sel.values())
            grd_total = sum(grd.values())
            return {
                "selected": sel,
                "skipped": skp,
                "grounded": grd,
                "selected_total": sel_total,
                "grounded_total": grd_total,
                "grounding_rate": (grd_total / sel_total) if sel_total > 0 else None,
            }

        r1 = round_block("R1")
        r2 = round_block("R2")
        totals_sel = {l: r1["selected"][l] + r2["selected"][l] for l in LEVELS}
        totals_skp = {l: r1["skipped"][l] + r2["skipped"][l] for l in LEVELS}
        totals_grd = {l: totals_sel[l] - totals_skp[l] for l in LEVELS}
        sel_sum = sum(totals_sel.values())
        grd_sum = sum(totals_grd.values())

        payload = {
            "mask_level": mask_level,
            "counting": "distinct_clues_at_most_specific_level",
            "level_specificity_order": ["street level clue", "city level clue", "country level clue"],
            "round1": r1,
            "round2": r2,
            "total": {
                "selected": totals_sel,
                "skipped": totals_skp,
                "grounded": totals_grd,
                "selected_total": sel_sum,
                "grounded_total": grd_sum,
                "grounding_rate": (grd_sum / sel_sum) if sel_sum > 0 else None,
            },
        }

        out_path = self.dirs['evaluation_results'] / 'clue_stats.json'
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[Pipeline] Saved clue stats to {out_path}")

    def _save_verification(self):
        """
        Write the granularity verdict for the run, and print it.

        This is Stage IV's output: not how far off the attacker was, but whether the cue
        granularities `experiment.mask_level` promised to remove are actually gone. Built from
        the verdicts already computed per image in `_create_evaluation_entry`, so it costs no
        extra API calls.
        """
        if not self._verdicts:
            return

        mask_level = self.config['experiment']['mask_level']
        report = RunVerification(
            privacy_class=self.spec.name,
            mask_level=mask_level,
            target_rung=target_rung(self.spec.name, mask_level),
            threshold_km=threshold_km(self.spec.name, mask_level),
            residual_source=str(self.dirs['eval_cache']),
            verdicts=dict(self._verdicts),
            matched=len(self._verdicts),
        )

        out_path = self.dirs['evaluation_results'] / 'verification.json'
        report.write(out_path)
        print(report.render())
        print(f"[Pipeline] Saved verification to {out_path}")

    def _print_timing_summary(self, total_time: float):
        """
        Print timing summary showing how long each phase took.

        Args:
            total_time: Total pipeline execution time in seconds
        """
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)

        # Print individual phase timings
        for phase_name, duration in self.phase_timings.items():
            minutes = int(duration // 60)
            seconds = duration % 60
            percentage = (duration / total_time * 100) if total_time > 0 else 0

            if minutes > 0:
                print(f"\n{phase_name}:")
                print(f"  Time: {minutes}m {seconds:.2f}s ({percentage:.1f}% of total)")
            else:
                print(f"\n{phase_name}:")
                print(f"  Time: {seconds:.2f}s ({percentage:.1f}% of total)")

        # Print total time
        print(f"\n{'=' * 60}")
        total_minutes = int(total_time // 60)
        total_seconds = total_time % 60

        if total_minutes > 0:
            print(f"Total Pipeline Time: {total_minutes}m {total_seconds:.2f}s")
        else:
            print(f"Total Pipeline Time: {total_seconds:.2f}s")

        # Convert to hours if > 60 minutes
        if total_minutes >= 60:
            hours = total_minutes // 60
            remaining_minutes = total_minutes % 60
            print(f"                     ({hours}h {remaining_minutes}m {total_seconds:.2f}s)")

        print("=" * 60)

    # ========================================================================
    # Main Execution
    # ========================================================================

    async def run(self):
        """
        Run the complete pipeline with all phases.
        """
        # Track total pipeline time
        pipeline_start = time.time()

        # Phase 1: Dataset loading
        phase1_start = time.time()
        self.load_dataset()
        phase1_time = time.time() - phase1_start
        self.phase_timings['Phase 1 (Dataset Loading)'] = phase1_time

        # Phase 2-4: Two-round evaluation and clue removal
        phase2_start = time.time()
        await self.run_parallel_pipeline()
        phase2_time = time.time() - phase2_start
        self.phase_timings['Phase 2-4 (Two-Round Processing)'] = phase2_time

        # Phase 5: Evaluation and reporting
        phase5_start = time.time()
        await self.evaluate_processed_images()
        phase5_time = time.time() - phase5_start
        self.phase_timings['Phase 5 (Evaluation & Reporting)'] = phase5_time

        # Calculate total time
        total_time = time.time() - pipeline_start

        # Display comprehensive timing summary
        self._print_timing_summary(total_time)

        # Display per-mask-level clue accounting (no-op outside geolocation spec)
        self._print_clue_stats()
        self._save_clue_stats()

        # A run that grounded nothing edited nothing; refuse to call that success.
        self.clue_stats.assert_something_was_grounded()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)
