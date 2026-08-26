"""
API Queue Handler for Parallel Pipeline

This module provides async workers that consume from the API queue,
process GPT evaluation requests, and dispatch follow-up tasks.

Workflow:
- R1_EVAL: Call GPT → Cache result → Update state → Push to seg_queue
- R2_EVAL: Call GPT → Cache result → Update state → Check 5-complete → Dispatch seg tasks
- FINAL_EVAL: Call GPT → Cache result → Update state → Increment complete counter
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2

from veillens.loop.queues import (
    APITask,
    QueueManager,
    TaskType,
    create_final_eval_task,
    create_seg_task,
)
from veillens.loop.state import GlobalState, ImageState
from veillens.privacy.spec import get_spec

logger = logging.getLogger(__name__)


class APIQueueHandler:
    """
    Async handler for API queue tasks.

    Consumes tasks from api_queue, makes GPT API calls,
    caches results, updates global state, and dispatches
    follow-up tasks to seg_queue.

    Attributes:
        queue_manager: QueueManager instance
        global_state: GlobalState instance
        geo_api: GeoLocationAPI instance
        config: Pipeline configuration
        output_base: Base output directory for caching
    """

    def __init__(
        self,
        queue_manager: QueueManager,
        global_state: GlobalState,
        geo_api,  # GeoLocationAPI instance
        config: Dict[str, Any]
    ):
        """
        Initialize API queue handler.

        Args:
            queue_manager: QueueManager instance
            global_state: GlobalState instance
            geo_api: GeoLocationAPI instance
            config: Pipeline configuration dict
        """
        self.qm = queue_manager
        self.gs = global_state
        self.api = geo_api
        self.config = config
        self.spec = get_spec(config)

        # The iteration gate is the defense's own control loop, not the reported metric, so it
        # runs on the reasoning backend -- the same oracle that produced the clues it is checking.
        self.gate_role = 'reason'

        # Output paths
        self.output_base = Path(config['output']['base_dir'])
        self.original_cache_dir = self.output_base / "original_cache"
        self.eval_cache_dir = self.output_base / "eval_cache"

        # Ensure directories exist
        self.original_cache_dir.mkdir(parents=True, exist_ok=True)
        self.eval_cache_dir.mkdir(parents=True, exist_ok=True)

        # Track worker ID for logging
        self._worker_id = 0

    async def run(self, worker_id: int = 0):
        """
        Main worker loop - runs until shutdown.

        Continuously pulls tasks from api_queue and processes them.

        Args:
            worker_id: Identifier for this worker (for logging)
        """
        self._worker_id = worker_id
        logger.info(f"[API Worker {worker_id}] Starting")

        while not self.qm.is_shutdown():
            try:
                # Get next task with timeout to check shutdown
                task = await asyncio.wait_for(
                    self.qm.get_api_task(),
                    timeout=1.0
                )

                try:
                    await self._process_task(task)
                except Exception as e:
                    logger.error(f"[API Worker {worker_id}] Error processing {task}: {e}")
                finally:
                    self.qm.api_task_done()

            except asyncio.TimeoutError:
                # Check shutdown flag and continue
                continue
            except asyncio.CancelledError:
                logger.info(f"[API Worker {worker_id}] Cancelled")
                break

        logger.info(f"[API Worker {worker_id}] Stopped")

    async def _process_task(self, task: APITask):
        """
        Process a single API task.

        Routes to appropriate handler based on task type.
        """
        logger.info(f"[API Worker {self._worker_id}] Processing {task}")

        if task.task_type == TaskType.R1_EVAL:
            await self._process_r1_eval(task)
        elif task.task_type == TaskType.R2_EVAL:
            await self._process_r2_eval(task)
        elif task.task_type == TaskType.GATE_EVAL:
            await self._process_gate_eval(task)
        elif task.task_type == TaskType.FINAL_EVAL:
            await self._process_final_eval(task)
        else:
            logger.error(f"Unknown task type: {task.task_type}")

    # =========================================================================
    # R1 Evaluation Handler
    # =========================================================================

    async def _process_r1_eval(self, task: APITask):
        """
        Process Round 1 evaluation task.

        1. Check external cache if configured
        2. If not cached, call GPT API with 'original' schema
        3. Tag features as 'across region'
        4. Cache result to original_cache
        5. Update state with prediction path and location
        6. Push segmentation task to seg_queue
        """
        state = self.gs.get_image(task.image_id)
        if not state:
            logger.error(f"No state for image {task.image_id}")
            return

        result = await self.api.get_geolocation_prediction(
            task.image_path,
            schema_type='original',
            role='reason',
        )

        if not result:
            # Mark complete rather than dropping the image. The driver's main loop runs until
            # num_complete == total_images, so an image that leaves the state machine without
            # being counted spins that loop until walltime -- the whole job then does nothing
            # for hours. R2 and the final eval already guard against this; R1 did not, and one
            # truncated response was enough to hang a 4-image run.
            logger.error(
                f"[R1 Eval] Failed for {task.image_id} -- no prediction, so there is nothing to "
                f"mask. Marking complete and leaving the image unprotected. See stdout for the "
                f"backend's reason."
            )
            state = self.gs.get_image(task.image_id)
            if state is not None:
                state.final_eval = "FAILED_R1"
            self.gs.increment_complete()
            return

        # Tag features as "across region"
        for feature in result.get('critical_features', {}).get('features', []):
            feature['region'] = 'across region'

        # Build R2 context string from R1 result (privacy-class-aware)
        predicted_location = self.spec.format_round1_context(result)

        # Cache result (always save to local cache)
        cache_path = self._save_prediction(
            task.image_id,
            state.image_name,
            "round1_full_img",
            result
        )

        # Update state
        state.r1_pred_path = str(cache_path)
        state.init_pred_loc = predicted_location

        logger.info(
            f"[R1 Eval] {task.image_id}: "
            f"{len(result.get('critical_features', {}).get('features', []))} features, "
            f"location: {predicted_location[:50]}..."
        )

        # Push segmentation task
        seg_task = create_seg_task(
            image_id=task.image_id,
            region="full_img",
            image_path=task.image_path,
            pred_path=str(cache_path)
        )
        self.qm.put_seg_task(seg_task)
        logger.debug(f"[R1 Eval] Pushed seg task for {task.image_id}")

    # =========================================================================
    # R2 Evaluation Handler
    # =========================================================================

    async def _process_r2_eval(self, task: APITask):
        """
        Process Round 2 regional evaluation task.

        1. Call GPT API with 'original' schema and R1 context
        2. Tag features with region name
        3. Cache result to original_cache
        4. Update state, track R2 prediction count
        5. When all 5 regions received, push seg tasks for each
        """
        state = self.gs.get_image(task.image_id)
        if not state:
            logger.error(f"No state for image {task.image_id}")
            return

        region = task.region
        if not region:
            logger.error(f"[R2 Eval] No region specified for {task.image_id}")
            return

        # Call GPT API with R1 context
        result = await self.api.get_geolocation_prediction(
            task.image_path,
            schema_type='original',
            predicted_location=state.init_pred_loc,
            region_name=region,
            cache_key_override=state.image_name,
            role='reason'
        )

        if not result:
            logger.error(f"[R2 Eval] Failed for {task.image_id} {region}")
            # Still need to track that this region was attempted
            # so we don't hang waiting for 5 predictions
            state.add_r2_pred_path(f"FAILED_{region}")
            return

        # Tag features with region
        for feature in result.get('critical_features', {}).get('features', []):
            feature['region'] = region

        # Cache result (iter-prefixed: iter{i}_round2_{region}.json)
        region_safe = region.replace(' ', '_')
        cache_path = self._save_prediction(
            task.image_id,
            state.image_name,
            f"iter{task.iter_idx}_round2_{region_safe}",
            result
        )

        # Update state
        num_received = state.add_r2_pred_path(str(cache_path))

        logger.info(
            f"[R2 Eval iter{task.iter_idx}] {task.image_id} {region}: "
            f"{len(result.get('critical_features', {}).get('features', []))} features "
            f"({num_received}/{state.expected_r2_regions} regions)"
        )

        # Dispatch seg task immediately for this region (don't wait for all!)
        await self._dispatch_r2_seg_task_for_region(state, region, str(cache_path), task.iter_idx)

    async def _dispatch_r2_seg_task_for_region(self, state: ImageState, region: str, pred_path: str, iter_idx: int = 1):
        """
        Dispatch segmentation task for a single R2 region immediately after API response.

        This enables better parallelism - segmentation can start on region 0 while
        API workers are still evaluating other regions.

        Args:
            state: ImageState instance
            region: Region name (integer string like "0", "1", "2", ...)
            pred_path: Path to cached prediction JSON for this region
            iter_idx: Round 2 iteration index this region belongs to
        """
        # Get crop path (iter-specific crop for this region)
        crop_path = self._get_r1_crop_path(state.image_id, state.image_name, region, iter_idx)

        # Push seg task
        seg_task = create_seg_task(
            image_id=state.image_id,
            region=region,
            image_path=crop_path,
            pred_path=pred_path,
            iter_idx=iter_idx,
        )
        self.qm.put_seg_task(seg_task)
        logger.debug(f"[R2 Seg Dispatch iter{iter_idx}] Pushed seg task for {state.image_id} {region}")

    async def _dispatch_r2_seg_tasks(self, state: ImageState):
        """
        DEPRECATED: Legacy method that dispatched all regions at once.

        Now using _dispatch_r2_seg_task_for_region() to dispatch immediately
        after each region's API response.

        Kept for reference but no longer called.
        """
        # Iterate over actual r2_pred_paths instead of hardcoded region list
        for pred_path in state.r2_pred_paths:
            if "FAILED" in pred_path:
                continue

            # Extract region name from path (e.g., "round2_0.json" -> "0")
            import re
            match = re.search(r'round2_(\d+)\.json', pred_path)
            if not match:
                logger.warning(f"[R2 Seg Dispatch] Could not parse region from path: {pred_path}")
                continue

            region = match.group(1)

            # Get crop path (R1 final image crop for this region)
            crop_path = self._get_r1_crop_path(state.image_id, state.image_name, region)

            # Push seg task
            seg_task = create_seg_task(
                image_id=state.image_id,
                region=region,
                image_path=crop_path,
                pred_path=pred_path
            )
            self.qm.put_seg_task(seg_task)
            logger.debug(f"[R2 Seg Dispatch] Pushed seg task for {state.image_id} {region}")

    def _get_r1_crop_path(self, image_id: str, image_name: str, region: str, iter_idx: int = 1) -> str:
        """
        Get path to iter-specific regional crop for R2 evaluation.

        Crops are saved by the segmentation handler under
        intermediate/{image_name}/r1_crops/iter{iter_idx}/{image_name}_{region}.png.

        Args:
            image_id: Image identifier
            image_name: Image filename stem
            region: Region name (integer string like "0", "1", "2", ...)
            iter_idx: Round 2 iteration index

        Returns:
            Path to regional crop image
        """
        crop_dir = self.output_base / "intermediate" / image_name / "r1_crops" / f"iter{iter_idx}"
        return str(crop_dir / f"{image_name}_{region}.png")

    # =========================================================================
    # Per-Iteration Gate Eval Handler
    # =========================================================================

    async def _process_gate_eval(self, task: APITask):
        """
        Process the per-iteration full-image gate evaluation.

        For iter i, runs a full-image GPT prediction on the image entering iter i
        (gray-masked R1 image for iter 1; working image for iter >=2). Caches as
        iter{i}_full_img.json. If the prediction's location error >= val_dist (km),
        marks the image as "stop after current iter" — the compositor will copy the
        in-flight working image to final.png once outstanding SD work for iter (i-1)
        completes (or immediately if no SD work is pending). Otherwise pushes an
        r2_setup SegTask to begin iter i's regional R2.
        """
        state = self.gs.get_image(task.image_id)
        if not state:
            logger.error(f"No state for image {task.image_id}")
            return

        iter_idx = task.iter_idx

        # Full-image GPT eval (same call shape as R1)
        result = await self.api.get_geolocation_prediction(
            task.image_path,
            schema_type='original',
            role=self.gate_role
        )

        if not result:
            logger.error(f"[Gate iter{iter_idx}] Failed for {task.image_id}; conservatively continuing")
            # On failure: skip gate decision, continue into R2 with last-known prediction
            seg_task = create_seg_task(
                image_id=task.image_id,
                region="r2_setup",
                image_path=task.image_path,
                pred_path=state.r1_pred_path or "",
                iter_idx=iter_idx,
            )
            self.qm.put_seg_task(seg_task)
            return

        # Tag features 'across region' (consistent with R1 cache shape — downstream
        # R2 only filters by mask_level so the tag is informational).
        for feature in result.get('critical_features', {}).get('features', []):
            feature['region'] = 'across region'

        # Cache iter{i}_full_img.json
        cache_path = self._save_prediction(
            task.image_id,
            state.image_name,
            f"iter{iter_idx}_full_img",
            result
        )

        # Compute error using privacy-class-aware metric.
        # For geolocation: haversine km. For personal_info: |score diff|.
        err = None
        if state.ground_truth is not None:
            try:
                gt = self.spec.normalize_ground_truth(state.ground_truth)
                err = self.spec.compute_error(gt, result)
            except Exception as e:
                logger.warning(f"[Gate iter{iter_idx}] {task.image_id}: error metric failed: {e}")

        val_dist = float(self.config['experiment'].get('val_dist', float('inf')))
        # Only apply the val_dist gate when the metric is a geolocation distance (km).
        # For personal_info the metric is a unitless score diff; never stop early.
        should_stop = (
            self.spec.name == 'geolocation'
            and err is not None
            and err >= val_dist
        )

        logger.info(
            f"[Gate iter{iter_idx}] {task.image_id}: err={err}, "
            f"val_dist={val_dist}, stop={should_stop}"
        )

        if should_stop:
            state.stop_after_current_iter = True
            # No R2 work will be queued for this iter. Mark r2_seg_complete=True
            # via set_expected_r2_regions(0) so is_complete() correctly reflects
            # "only SD catch-up pending". Without this, is_complete() returns False
            # and nothing ever finalizes the image (deadlock).
            state.set_expected_r2_regions(0)
            if state.is_complete():
                # Prefer the working image (post-SD result) when it exists; only
                # fall back to task.image_path (the iter source) if no SD has run.
                working_path = (
                    self.output_base / "intermediate" / state.image_name
                    / "working" / f"{state.image_name}_working.png"
                )
                source = str(working_path) if working_path.exists() else task.image_path
                await self._finalize_and_push_final_eval(state, source)
            return

        # Continue: refresh R2 context for downstream R2_EVAL calls
        state.init_pred_loc = self.spec.format_round1_context(result)

        fine_grained = self.config.get('round2_regions', {}).get('fine_grained', False)

        if fine_grained:
            # Regional crops: extract patches and dispatch one R2_EVAL per crop.
            seg_task = create_seg_task(
                image_id=task.image_id,
                region="r2_setup",
                image_path=task.image_path,  # iter source: the protected image this gate read
                pred_path=str(cache_path),
                iter_idx=iter_idx,
            )
            self.qm.put_seg_task(seg_task)
            logger.debug(f"[Gate iter{iter_idx}] {task.image_id}: pushed r2_setup")
        else:
            # Full-image R2: reuse the gate-eval cache as the R2 prediction; no extra GPT call.
            # _process_r2_setup and per-region R2_EVAL are both skipped, so:
            #  - set expected_r2_regions here (single full-image "region")
            #  - add the gate-eval cache as the sole r2_pred_path so is_complete()'s
            #    `len(r2_pred_paths) == expected_r2_regions` gate fires
            # Clear last iter's clue budget. This used to be skipped for iter 1,
            # whose budget still held the R1 clues; R1 is iteration 0 now, so it has
            # been accounted for and every iteration starts from a clean count.
            # (The advancing iteration already reset, making this a no-op there.)
            state.reset_for_next_iter()
            state.r2_iter = iter_idx
            state.set_expected_r2_regions(1)
            state.add_r2_pred_path(str(cache_path))
            seg_task = create_seg_task(
                image_id=task.image_id,
                region="r2_full_img",
                image_path=task.image_path,
                pred_path=str(cache_path),
                iter_idx=iter_idx,
            )
            self.qm.put_seg_task(seg_task)
            logger.debug(f"[Gate iter{iter_idx}] {task.image_id}: pushed r2_full_img")

    async def _finalize_and_push_final_eval(self, state: ImageState, source_image_path: str):
        """
        Copy source image to canonical final.png and push FINAL_EVAL (or skip if
        evaluation disabled). Used when the gate decides to stop and there is no
        in-flight SD work pending.
        """

        final_dir = self.output_base / "intermediate" / state.image_name / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / f"{state.image_name}_final.png"

        # Both failures below are terminal for this image, so it has to be counted on the way out.
        # The main loop runs until num_complete == total_images; an image that leaves the state
        # machine uncounted spins that loop until walltime. Same bug class as the R1 path above.
        try:
            img = cv2.imread(source_image_path)
            if img is None:
                logger.error(
                    f"[Gate finalize] {state.image_id}: failed to read {source_image_path} -- "
                    f"marking complete with no final image."
                )
                state.final_eval = "FAILED_FINALIZE"
                self.gs.increment_complete()
                return
            cv2.imwrite(str(final_path), img)
        except Exception as e:
            logger.error(
                f"[Gate finalize] {state.image_id}: write failed: {e} -- marking complete."
            )
            state.final_eval = "FAILED_FINALIZE"
            self.gs.increment_complete()
            return

        state.final_image = str(final_path)

        task = create_final_eval_task(state.image_id, str(final_path))
        await self.qm.put_api_task(task)
        logger.info(f"[Gate finalize] {state.image_id}: pushed final_eval (early stop)")

    # =========================================================================
    # Final Evaluation Handler
    # =========================================================================

    async def _process_final_eval(self, task: APITask):
        """
        Process final evaluation task.

        1. Call GPT API with 'eval' schema
        2. Cache result to eval_cache
        3. Update state
        4. Increment global completion counter
        """
        state = self.gs.get_image(task.image_id)
        if not state:
            logger.error(f"No state for image {task.image_id}")
            return

        # Call GPT API with eval schema. For personal_info, the same schema is
        # reused for the final eval (it already exposes the privacy score).
        eval_schema_type = 'eval' if self.spec.name == 'geolocation' else 'original'
        result = await self.api.get_geolocation_prediction(
            task.image_path,
            schema_type=eval_schema_type,
            role='eval'
        )

        if not result:
            logger.error(f"[Final Eval] Failed for {task.image_id}")
            # Still mark complete to avoid hanging
            state.final_eval = "FAILED"
            self.gs.increment_complete()
            return

        # Cache result to eval_cache
        cache_path = self.eval_cache_dir / f"{state.image_name}_eval.json"
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Update state
        state.final_eval = str(cache_path)

        # Increment completion counter
        num_complete = self.gs.increment_complete()
        completed, total = self.gs.get_progress()

        logger.info(f"[Final Eval] {task.image_id}: Complete ({completed}/{total})")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _save_prediction(
        self,
        image_id: str,
        image_name: str,
        filename: str,
        result: Dict
    ) -> Path:
        """
        Save prediction result to original_cache.

        Args:
            image_id: Image identifier
            image_name: Image filename stem
            filename: Cache filename (without extension)
            result: Prediction result dict

        Returns:
            Path to saved cache file
        """
        cache_dir = self.original_cache_dir / image_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = cache_dir / f"{filename}.json"
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        return cache_path


async def create_api_workers(
    queue_manager: QueueManager,
    global_state: GlobalState,
    geo_api,
    config: Dict[str, Any],
    num_workers: int = 3
) -> List[asyncio.Task]:
    """
    Create and start API worker tasks.

    Args:
        queue_manager: QueueManager instance
        global_state: GlobalState instance
        geo_api: GeoLocationAPI instance
        config: Pipeline configuration
        num_workers: Number of concurrent workers

    Returns:
        List of asyncio.Task objects for the workers
    """
    handler = APIQueueHandler(queue_manager, global_state, geo_api, config)

    tasks = []
    for i in range(num_workers):
        task = asyncio.create_task(handler.run(worker_id=i))
        tasks.append(task)

    return tasks
