"""
Segmentation Queue Handler for Parallel Pipeline

This module provides the GPU worker that consumes from the segmentation queue,
performs Grounded SAM 2 segmentation, and dispatches follow-up tasks.

Runs in a separate thread (GPU-bound, single-threaded).

Workflow:
- R1 Segmentation: Extract depth → Segment → Gray-mask image → Push R2 eval tasks → Push SD tasks
- R2 Segmentation: Segment → Push SD tasks → Track R2 completion
"""

import asyncio
import json
import logging
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from veillens.loop.batching import ClueMetadata, calculate_mask_bbox, classify_clue
from veillens.loop.queues import (
    QueueManager,
    SegTask,
    create_final_eval_task,
    create_gate_eval_task,
    create_r2_eval_task,
)
from veillens.loop.state import ClueStatsTracker, GlobalState, ImageState
from veillens.privacy.spec import get_spec
from veillens.paths import FINAL_DIR, INTERMEDIATE_DIR, SEGMENTATION_DIR

logger = logging.getLogger(__name__)


# Note: Sliding window parameters are loaded from config['round2_regions']
# Region names are now integer strings ("0", "1", "2", ...) instead of position names


class SegmentationQueueHandler:
    """
    GPU worker for segmentation queue tasks.

    Consumes tasks from seg_queue, runs Grounded SAM 2 segmentation,
    creates masks, and dispatches follow-up tasks.

    Runs synchronously in a dedicated thread (GPU is single-threaded).

    Attributes:
        queue_manager: QueueManager instance
        global_state: GlobalState instance
        sdxl_pipeline: SDXLInpaintingPipeline instance
        config: Pipeline configuration
        gray_color: Color for masking (neutral gray 128,128,128)
    """

    def __init__(
        self,
        queue_manager: QueueManager,
        global_state: GlobalState,
        sdxl_pipeline,  # SDXLInpaintingPipeline instance
        config: Dict[str, Any],
        event_loop: asyncio.AbstractEventLoop,
        clue_stats: Optional[ClueStatsTracker] = None
    ):
        """
        Initialize segmentation queue handler.

        Args:
            queue_manager: QueueManager instance
            global_state: GlobalState instance
            sdxl_pipeline: SDXLInpaintingPipeline instance
            config: Pipeline configuration dict
            event_loop: Main asyncio event loop (for pushing async tasks)
            clue_stats: Optional ClueStatsTracker for end-of-run accounting
        """
        self.qm = queue_manager
        self.gs = global_state
        self.pipeline = sdxl_pipeline
        self.config = config
        self.event_loop = event_loop
        self.clue_stats = clue_stats

        # Output paths
        self.output_base = Path(config['output']['base_dir'])

        # Mask color (user confirmed: neutral gray)
        self.gray_color = (128, 128, 128)

        # Privacy class spec drives feature filtering and field naming
        self.spec = get_spec(config)
        self.mask_level = config['experiment']['mask_level']

    def run(self):
        """
        Main worker loop - runs until shutdown.

        Blocking loop that pulls tasks from seg_queue and processes them.
        Runs in a dedicated thread.
        """
        logger.info("[Seg Worker] Starting")

        # Enable float16 autocast for this thread (SAM2 requires it)
        # Autocast context is thread-local, so we need to enable it here
        device = self.config['detection']['device']
        if device == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
            logger.info("[Seg Worker] Float16 autocast enabled for segmentation thread")

        while not self.qm.is_shutdown():
            task = None
            try:
                # Get next task with timeout to check shutdown
                task = self.qm.get_seg_task(timeout=1.0)
                if task is None:
                    continue

                # Wait if SD is currently running (GPU coordination)
                if self.qm.is_sd_running():
                    logger.debug(f"[Seg Worker] Pausing - SD is running (task: {task})")

                while self.qm.is_sd_running():
                    if self.qm.is_shutdown():
                        break
                    time.sleep(0.5)  # Check every 500ms

                # If shutdown during wait, exit gracefully
                if self.qm.is_shutdown():
                    # Put task back if we didn't process it
                    if task is not None:
                        self.qm.seg_queue.put(task)
                    break

                if not self.qm.is_sd_running():
                    logger.debug(f"[Seg Worker] Resuming - SD finished (task: {task})")

                try:
                    self._process_task(task)
                except Exception as e:
                    logger.error(f"[Seg Worker] Error processing {task}: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"[Seg Worker] Unexpected error in main loop: {e}", exc_info=True)
            finally:
                # CRITICAL: Always mark task as done to prevent queue deadlock
                if task is not None:
                    self.qm.seg_task_done()

        logger.info("[Seg Worker] Stopped")

    def _process_task(self, task: SegTask):
        """
        Process a single segmentation task.

        Routes to R1 or R2 handler based on region.
        """
        logger.info(f"[Seg Worker] Processing {task}")

        if task.region == "full_img":
            self._process_r1_segmentation(task)
        elif task.region == "r2_setup":
            self._process_r2_setup(task)
        elif task.region == "r2_full_img":
            self._process_r2_full_image(task)
        else:
            self._process_r2_segmentation(task)

    # =========================================================================
    # R1 Segmentation Handler
    # =========================================================================

    def _process_r1_segmentation(self, task: SegTask):
        """
        Process Round 1 (full image) segmentation task.

        Steps:
        1. Load image
        2. Generate + cache the depth map
        3. Load prediction, filter features by mask level
        4. Segment all clues using Grounded SAM 2
        5. Create gray-masked R1 image
        6. Extract sliding window regional crops, push R2 eval tasks
        7. Push R1 clues to SD queue
        """
        state = self.gs.get_image(task.image_id)
        if not state:
            logger.error(f"No state for image {task.image_id}")
            return

        # 1. Load image
        image = cv2.imread(task.image_path)
        if image is None:
            logger.error(f"[R1 Seg] Failed to load {task.image_path}")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # 2. Generate + cache the depth map
        depth_map = self._get_or_compute_depth(
            task.image_id, state.image_name, image_rgb
        )
        state.depth_path = str(self._get_depth_path(state.image_name))

        # 3. Load prediction, filter by mask level
        features = self._load_and_filter_features(task.pred_path)

        logger.info(f"[R1 Seg] {task.image_id}: {len(features)} features to segment")

        if self.clue_stats is not None:
            for feature in features:
                self.clue_stats.record_selected("R1", feature)

        # 4. Segment all clues
        masks = []
        mask_paths = []
        if features:
            for idx, feature in enumerate(features):
                object_desc = feature.get('object_description', '')
                target_concept = feature.get('target_concept', None)

                # Segment using Grounded SAM 2
                try:
                    _, seg_masks, boxes, labels = self.pipeline.segmenter.segment(
                        task.image_path, object_desc, target_concept
                    )

                    if len(seg_masks) == 0:
                        logger.warning(f"[R1 Seg] No mask for: {object_desc[:50]}")
                        if self.clue_stats is not None:
                            self.clue_stats.record_skipped("R1", feature)
                        continue

                    # Combine masks
                    combined_mask = self.pipeline.segmenter.combine_masks(seg_masks)
                    masks.append(combined_mask)

                    # Save mask
                    mask_path = self._save_mask(
                        state.image_name, "full_img", object_desc, combined_mask
                    )
                    mask_paths.append(str(mask_path))
                    state.add_mask_path(str(mask_path))

                    logger.debug(f"[R1 Seg] Segmented: {object_desc[:30]}...")

                except Exception as e:
                    logger.error(f"[R1 Seg] Segmentation failed for {object_desc[:30]}: {e}")
                    if self.clue_stats is not None:
                        self.clue_stats.record_skipped("R1", feature)

        # 5. Save the gray-masked R1 image. Diagnostic only: it shows at a glance what
        # R1 decided to remove. The gate reads the inpainted result instead (step 7),
        # so the next round attacks a real protected image, not a gray-boxed one.
        if masks:
            r1_image = self._create_gray_masked_image(image_rgb, masks)
        else:
            # Zero clues: copy original (user choice: still run R2)
            r1_image = image_rgb.copy()
            logger.info(f"[R1 Seg] {task.image_id}: No clues found, using original for R2")

        r1_path = self._save_r1_image(state.image_name, r1_image)
        state.r1_final_image = r1_path

        # 6. R1 is its own iteration, numbered 0: its clues are the only work in it.
        # Declaring zero R2 regions marks r2_seg_complete, which is what lets
        # is_complete() fire once the R1 clues are composited -- the compositor then
        # snapshots final/iter0_<name>_final.png and pushes the gate on it.
        state.r2_iter = 0
        state.set_expected_r2_regions(0)
        state.set_num_total_clue(len(masks))

        if features and masks:
            self._push_clues_to_sd_queue(
                task.image_id,
                state.image_name,
                features[:len(masks)],  # Match features to successful masks
                masks,
                mask_paths,
                image_rgb,  # Original image for SD inpainting
                depth_map,
                region_name=None,  # R1
                iter_idx=0,
            )
            logger.info(f"[R1 Seg] {task.image_id}: Complete, {len(masks)} masks queued as iter 0")
        else:
            # Nothing to inpaint, so no SD item will ever trigger the compositor.
            # Snapshot and advance here or the image would sit in the queue forever.
            logger.info(f"[R1 Seg] {task.image_id}: 0 masks, advancing from iter 0 directly")
            self._advance_iter_or_finalize_disk(state, task.image_path, 0)

    # =========================================================================
    # R2 Setup Handler (per-iteration crop extraction + R2_EVAL dispatch)
    # =========================================================================

    def _process_r2_setup(self, task: SegTask):
        """
        Set up Round 2 for iteration `task.iter_idx`.

        Called after the gate eval votes "continue". Extracts sliding-window crops
        from the iter source image, sets expected_r2_regions, and pushes one
        R2_EVAL task per crop. For iter >=2 also resets per-iteration R2 counters
        on the ImageState before setting expected_r2_regions so is_complete() fires
        on the right cadence.

        Edge cases:
        - len(crops) == 0 (image smaller than crop): no R2 work to do this iter.
          Treat as iter completion with zero R2 clues — advance or finalize.
        """
        state = self.gs.get_image(task.image_id)
        if not state:
            logger.error(f"No state for image {task.image_id}")
            return

        iter_idx = task.iter_idx

        # Load iter source image (gray-masked R1 for iter 1; working for iter >=2)
        full_image = cv2.imread(task.image_path)
        if full_image is None:
            logger.error(f"[R2 Setup iter{iter_idx}] Failed to load {task.image_path}")
            return
        image_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

        # Reset per-iter R2 state for iter >=2 BEFORE setting expected_r2_regions.
        # Iter 1 keeps its R1 num_total_clue baseline so is_complete fires only when
        # both R1 SD and iter-1 R2 SD are done.
        if iter_idx >= 2:
            state.reset_for_next_iter()
        state.r2_iter = iter_idx

        # Extract crops + write iter-specific dir
        crops = self._extract_and_save_regional_crops(state.image_name, image_rgb, iter_idx=iter_idx)
        state.set_expected_r2_regions(len(crops))
        logger.info(f"[R2 Setup iter{iter_idx}] {task.image_id}: {len(crops)} regions")

        if len(crops) == 0:
            # No R2 regions this iter → treat as iter completion with no R2 clues.
            # For iter 1 this commonly means the image is smaller than the crop window;
            # advance to next iter or finalize, depending on max_iter / stop flag.
            self._advance_iter_or_finalize_disk(state, task.image_path, iter_idx)
            return

        # Push R2_EVAL task per crop. iter_idx threads through to cache filename
        # and the SegTask returned by _dispatch_r2_seg_task_for_region.
        for region, crop_path in crops.items():
            api_task = create_r2_eval_task(task.image_id, crop_path, region, iter_idx=iter_idx)
            asyncio.run_coroutine_threadsafe(
                self.qm.put_api_task(api_task),
                self.event_loop
            )
            logger.debug(f"[R2 Setup iter{iter_idx}] Pushed R2 eval: {task.image_id} {region}")

    # =========================================================================
    # R2 Full-Image Segmentation Handler (fine_grained=false)
    # =========================================================================

    def _process_r2_full_image(self, task: SegTask):
        """
        Process Round 2 segmentation on the full image (no regional cropping).

        Used when config['round2_regions']['fine_grained'] is False. The gate-eval
        prediction (iter{i}_full_img.json) is reused as the R2 clue list; per-region
        crop extraction and per-region R2_EVAL GPT calls are skipped entirely.
        expected_r2_regions=1 was set by _process_gate_eval before dispatch.
        """
        state = self.gs.get_image(task.image_id)
        if not state:
            logger.error(f"No state for image {task.image_id}")
            return

        iter_idx = task.iter_idx
        region = "r2_full"  # mask dir: segmentation/iter{i}_r2_full/

        # 1. Load iter source image (gray-masked R1 for iter 1; working for iter >=2).
        # task.image_path was set by _process_gate_eval to the iter source already.
        iter_source_path = task.image_path
        full_image = cv2.imread(iter_source_path)
        if full_image is None:
            logger.error(f"[R2 FullImg iter{iter_idx}] Failed to load iter source {iter_source_path}")
            return
        full_image_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        h, w = full_image_rgb.shape[:2]

        # 2. Load prediction (gate-eval cache) and filter by mask level
        features = self._load_and_filter_features(task.pred_path)
        logger.info(f"[R2 FullImg iter{iter_idx}] {task.image_id}: {len(features)} features to segment")

        if self.clue_stats is not None:
            for feature in features:
                self.clue_stats.record_selected("R2", feature)

        # 3. Get depth for full image (cached from R1)
        depth_map = self._get_or_compute_depth(
            task.image_id, state.image_name, full_image_rgb
        )

        # 4. Segment each clue on the full image (no upscale/downscale/positioning).
        # Save the iter source to a temp file once; the segmenter takes a path.
        masks_full = []
        mask_paths = []
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, full_image)

        try:
            for feature in features:
                object_desc = feature.get('object_description', '')
                target_concept = feature.get('target_concept', None)

                try:
                    _, seg_masks, boxes, labels = self.pipeline.segmenter.segment(
                        temp_path, object_desc, target_concept
                    )

                    if len(seg_masks) == 0:
                        logger.warning(f"[R2 FullImg] No mask for: {object_desc[:50]}")
                        if self.clue_stats is not None:
                            self.clue_stats.record_skipped("R2", feature)
                        continue

                    combined_mask = self.pipeline.segmenter.combine_masks(seg_masks)
                    # Segmenter returns mask at the input image size — already full-image.
                    if combined_mask.shape[:2] != (h, w):
                        combined_mask = cv2.resize(
                            combined_mask, (w, h), interpolation=cv2.INTER_NEAREST
                        )
                    masks_full.append(combined_mask)

                    mask_path = self._save_mask(
                        state.image_name, region, object_desc, combined_mask, iter_idx=iter_idx
                    )
                    mask_paths.append(str(mask_path))
                    state.add_mask_path(str(mask_path))

                except Exception as e:
                    logger.error(f"[R2 FullImg] Segmentation failed for {object_desc[:30]}: {e}")
                    if self.clue_stats is not None:
                        self.clue_stats.record_skipped("R2", feature)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        if masks_full:
            state.add_to_total_clue(len(masks_full))

        # 5. Push to SD queue, inpainting on top of what earlier iterations produced.
        # Iter 1 used to special-case the original image because R1's inpaints had not
        # been composited yet; now R1 is iteration 0 and has, so every iteration edits
        # the accumulated working image.
        sd_source_path = self._current_image_path(state)
        sd_source_image = cv2.imread(sd_source_path)
        if sd_source_image is not None and features and masks_full:
            sd_source_rgb = cv2.cvtColor(sd_source_image, cv2.COLOR_BGR2RGB)
            self._push_clues_to_sd_queue(
                task.image_id,
                state.image_name,
                features[:len(masks_full)],
                masks_full,
                mask_paths,
                sd_source_rgb,
                depth_map,
                region_name=region,
                iter_idx=iter_idx,
            )

        # 6. Track R2 completion (single full-image task counts as one region)
        r2_count = state.increment_r2_seg_count()
        logger.info(
            f"[R2 FullImg iter{iter_idx}] {task.image_id}: Complete, "
            f"{len(masks_full)} masks ({r2_count}/{state.expected_r2_regions})"
        )

        # 7. Iter-completion: mirror the regional path's off-SD finalize logic
        # (zero-clue iter or SD-finished-before-seg race).
        if r2_count == state.expected_r2_regions and state.num_total_clue == 0:
            logger.info(f"[R2 FullImg iter{iter_idx}] {task.image_id}: 0 clues this iter, advancing")
            self._advance_iter_or_finalize_disk(state, self._current_image_path(state), iter_idx)
        elif r2_count == state.expected_r2_regions and state.is_complete():
            logger.info(f"[R2 FullImg iter{iter_idx}] {task.image_id}: race detected, advancing")
            self._advance_iter_or_finalize_disk(state, self._current_image_path(state), iter_idx)

    # =========================================================================
    # R2 Segmentation Handler
    # =========================================================================

    def _process_r2_segmentation(self, task: SegTask):
        """
        Process Round 2 (regional) segmentation task.

        Steps:
        1. Load iter source image (gray-masked R1 for iter 1; working image for iter >=2)
        2. Extract the specific region
        3. Load prediction, filter features by mask level
        4. Segment clues on upscaled region
        5. Downscale masks to original size, position in full image
        6. Push clues to SD queue (SD source: original for iter 1, working for iter >=2)
        7. Track R2 completion
        """
        state = self.gs.get_image(task.image_id)
        if not state:
            logger.error(f"No state for image {task.image_id}")
            return

        region = task.region
        iter_idx = task.iter_idx

        # 1. Load iter source image used for region extraction: the protected image as
        # of the previous iteration (iteration 0 having inpainted the R1 clues).
        iter_source_path = self._current_image_path(state)
        if not iter_source_path:
            logger.error(f"[R2 Seg iter{iter_idx}] No iter source image for {task.image_id}")
            return

        full_image = cv2.imread(iter_source_path)
        if full_image is None:
            logger.error(f"[R2 Seg iter{iter_idx}] Failed to load iter source {iter_source_path}")
            return
        full_image_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        h, w = full_image_rgb.shape[:2]

        # 2. Extract the specific region
        upscaled_region, crop_coords = self._extract_region(full_image_rgb, region)
        x1, y1, x2, y2 = crop_coords
        original_crop_w = x2 - x1
        original_crop_h = y2 - y1

        # 3. Load prediction, filter by mask level
        features = self._load_and_filter_features(task.pred_path)

        logger.info(f"[R2 Seg] {task.image_id} {region}: {len(features)} features to segment")

        if self.clue_stats is not None:
            for feature in features:
                self.clue_stats.record_selected("R2", feature)

        # Get depth for full image (should be cached from R1)
        depth_map = self._get_or_compute_depth(
            task.image_id, state.image_name, full_image_rgb
        )

        # 4. Segment clues on upscaled region
        masks_full = []
        mask_paths = []

        for idx, feature in enumerate(features):
            object_desc = feature.get('object_description', '')
            target_concept = feature.get('target_concept', None)

            # Save upscaled region to temp file for segmentation
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                upscaled_bgr = cv2.cvtColor(upscaled_region, cv2.COLOR_RGB2BGR)
                cv2.imwrite(temp_path, upscaled_bgr)

            try:
                # Segment on upscaled region
                _, seg_masks, boxes, labels = self.pipeline.segmenter.segment(
                    temp_path, object_desc, target_concept
                )

                if len(seg_masks) == 0:
                    logger.warning(f"[R2 Seg] No mask for: {object_desc[:50]}")
                    if self.clue_stats is not None:
                        self.clue_stats.record_skipped("R2", feature)
                    continue

                # Combine masks (at upscaled size)
                combined_mask_upscaled = self.pipeline.segmenter.combine_masks(seg_masks)

                # 5. Downscale mask to original crop size
                mask_downscaled = cv2.resize(
                    combined_mask_upscaled,
                    (original_crop_w, original_crop_h),
                    interpolation=cv2.INTER_NEAREST
                )

                # Position mask in full image coordinates
                mask_full = np.zeros((h, w), dtype=np.uint8)
                mask_full[y1:y2, x1:x2] = mask_downscaled
                masks_full.append(mask_full)

                # Save mask (iter-prefixed dir)
                mask_path = self._save_mask(
                    state.image_name, region, object_desc, mask_full, iter_idx=iter_idx
                )
                mask_paths.append(str(mask_path))
                state.add_mask_path(str(mask_path))

                logger.debug(f"[R2 Seg iter{iter_idx}] Segmented {region}: {object_desc[:30]}...")

            except Exception as e:
                logger.error(f"[R2 Seg] Segmentation failed for {object_desc[:30]}: {e}")
                if self.clue_stats is not None:
                    self.clue_stats.record_skipped("R2", feature)
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

        # Add to total clue count based on SUCCESSFUL segmentations
        if masks_full:
            state.add_to_total_clue(len(masks_full))

        # 6. Push clues to SD queue, editing the protected image the previous iteration
        # produced, so each iter refines the last result instead of restarting from the
        # original. (Iteration 0 is the R1 clues, so even iter 1 has a real predecessor.)
        sd_source_path = self._current_image_path(state)
        sd_source_image = cv2.imread(sd_source_path)
        if sd_source_image is not None:
            sd_source_rgb = cv2.cvtColor(sd_source_image, cv2.COLOR_BGR2RGB)

            if features and masks_full:
                self._push_clues_to_sd_queue(
                    task.image_id,
                    state.image_name,
                    features[:len(masks_full)],
                    masks_full,
                    mask_paths,
                    sd_source_rgb,
                    depth_map,
                    region_name=region,
                    iter_idx=iter_idx,
                )

        # 7. Track R2 completion
        r2_count = state.increment_r2_seg_count()
        logger.info(f"[R2 Seg iter{iter_idx}] {task.image_id} {region}: Complete, {len(masks_full)} masks ({r2_count}/{state.expected_r2_regions} regions)")

        # 8. Iter completion paths handled outside the normal SD composite flow.
        # Two cases the compositor cannot catch:
        #  (a) all R2 regions for this iter finished with zero clues → no SD task
        #      will fire, so is_complete() will never be observed by the compositor.
        #  (b) race: every SD task for this iter already completed before this last
        #      R2 region was segmented, so the compositor's is_complete() check
        #      saw r2_seg_complete=False and skipped finalization.
        # In both cases we run the same iter-advance/finalize logic the compositor
        # would have run, using the appropriate on-disk source.
        if r2_count == state.expected_r2_regions and state.num_total_clue == 0:
            # No clues this iter → the protected image is unchanged from iter start.
            logger.info(f"[R2 Seg iter{iter_idx}] {task.image_id}: 0 R2 clues this iter, advancing")
            self._advance_iter_or_finalize_disk(state, self._current_image_path(state), iter_idx)

        elif r2_count == state.expected_r2_regions and state.is_complete():
            # Race: SD finished before this last R2 seg ran. Working image is current.
            logger.info(f"[R2 Seg iter{iter_idx}] {task.image_id}: race detected, advancing")
            self._advance_iter_or_finalize_disk(state, self._current_image_path(state), iter_idx)

    def _advance_iter_or_finalize_disk(self, state: ImageState, current_image_path: str, iter_idx: int) -> None:
        """
        Mirror of the compositor's iter-advance logic for off-SD-path triggers.

        Writes the per-iter snapshot from current_image_path, then either pushes
        GATE_EVAL(iter+1) (advance) or FINAL_EVAL (terminate). Called from the seg
        worker thread; uses run_coroutine_threadsafe to enqueue async API tasks.
        """
        final_dir = self.output_base / INTERMEDIATE_DIR / state.image_name / FINAL_DIR
        final_dir.mkdir(parents=True, exist_ok=True)
        iter_snapshot = final_dir / f"iter{iter_idx}_{state.image_name}_final.png"

        img = cv2.imread(current_image_path)
        if img is None:
            logger.error(f"[Iter advance] Failed to read {current_image_path}")
            return
        cv2.imwrite(str(iter_snapshot), img)

        max_iter = int(self.config['experiment'].get('max_iter', 1))

        if (not state.stop_after_current_iter) and iter_idx < max_iter:
            # Advance: reset per-iter R2 state, bump iter, push next GATE_EVAL
            state.reset_for_next_iter()
            state.r2_iter = iter_idx + 1
            api_task = create_gate_eval_task(state.image_id, current_image_path, iter_idx=iter_idx + 1)
            asyncio.run_coroutine_threadsafe(
                self.qm.put_api_task(api_task),
                self.event_loop
            )
            logger.info(f"[Iter advance] {state.image_id}: pushed gate_eval(iter={iter_idx + 1})")
        else:
            # Finalize
            final_path = final_dir / f"{state.image_name}_final.png"
            cv2.imwrite(str(final_path), img)
            state.final_image = str(final_path)
            api_task = create_final_eval_task(state.image_id, str(final_path))
            asyncio.run_coroutine_threadsafe(
                self.qm.put_api_task(api_task),
                self.event_loop
            )
            logger.info(f"[Iter finalize] {state.image_id}: pushed final_eval after iter {iter_idx}")

    # =========================================================================
    # Utility Methods: Depth
    # =========================================================================

    def _get_or_compute_depth(
        self,
        image_id: str,
        image_name: str,
        image_rgb: np.ndarray
    ) -> np.ndarray:
        """
        Get the cached depth map or compute and cache it.

        Args:
            image_id: Image identifier
            image_name: Image filename stem
            image_rgb: Image array (RGB)

        Returns:
            Depth map (H, W) uint8
        """
        depth_path = self._get_depth_path(image_name)

        # Check cache
        if depth_path.exists():
            depth_map = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            if depth_map is not None:
                logger.debug(f"[Depth] Cache hit for {image_name}")
                return depth_map

        # Compute
        logger.debug(f"[Depth] Computing for {image_name}")
        depth_map = self.pipeline.extractor.extract_depth(image_rgb)

        # Cache
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(depth_path), depth_map)

        return depth_map

    def _get_depth_path(self, image_name: str) -> Path:
        """Get path for depth cache file."""
        return self.output_base / "depth" / f"{image_name}.png"

    # =========================================================================
    # Utility Methods: Feature Loading
    # =========================================================================

    def _load_and_filter_features(self, pred_path: str) -> List[Dict]:
        """
        Load prediction JSON and filter features by mask level.

        Args:
            pred_path: Path to prediction JSON

        Returns:
            List of filtered features
        """
        try:
            with open(pred_path, 'r') as f:
                prediction = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load prediction {pred_path}: {e}")
            return []

        features = prediction.get('critical_features', {}).get('features', [])

        # Filter by mask level (semantics depend on privacy class spec)
        filtered = [f for f in features if self.spec.feature_passes_mask_level(f, self.mask_level)]

        return filtered

    # =========================================================================
    # Utility Methods: Mask Saving
    # =========================================================================

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Sanitize object description for use in filenames."""
        # Remove invalid filename characters
        text = re.sub(r'[/\\:*?"<>|]', '', text)
        # Replace spaces with underscores
        text = text.replace(' ', '_')
        # Keep only alphanumeric, underscores, hyphens
        text = re.sub(r'[^a-zA-Z0-9_-]', '', text)
        # Limit length
        text = text[:max_length].rstrip('_-')
        return text if text else "unknown"

    def _save_mask(
        self,
        image_name: str,
        region: str,
        object_desc: str,
        mask: np.ndarray,
        iter_idx: int = 1,
    ) -> Path:
        """
        Save segmentation mask to disk.

        For R1 (region="full_img"), masks go to segmentation/full_img/.
        For R2 regions, masks go to segmentation/iter{iter_idx}_{region}/ so each
        iteration's masks live in their own directory.

        Args:
            image_name: Image filename stem
            region: Region name (full_img, "0", "1", ...)
            object_desc: Object description for filename
            mask: Binary mask array
            iter_idx: Round 2 iteration index (ignored for R1)

        Returns:
            Path to saved mask file
        """
        region_safe = region.replace(' ', '_')
        sanitized_desc = self._sanitize_filename(object_desc)

        if region == "full_img":
            sub = "full_img"
        else:
            sub = f"iter{iter_idx}_{region_safe}"
        seg_dir = self.output_base / INTERMEDIATE_DIR / image_name / SEGMENTATION_DIR / sub
        seg_dir.mkdir(parents=True, exist_ok=True)

        mask_path = seg_dir / f"{sanitized_desc}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        return mask_path

    # =========================================================================
    # Utility Methods: R1 Image Processing
    # =========================================================================

    def _create_gray_masked_image(
        self,
        image: np.ndarray,
        masks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Fill mask regions with neutral gray (128,128,128).

        Args:
            image: Original image (RGB)
            masks: List of binary masks

        Returns:
            Image with masked regions filled with gray
        """
        result = image.copy()
        for mask in masks:
            result[mask > 0] = self.gray_color
        return result

    def _current_image_path(self, state: ImageState) -> str:
        """
        The most protected version of this image that exists on disk.

        That is the working image, which the compositor updates after every inpaint.
        Before any inpaint has landed -- an image whose R1 clues all failed to ground --
        there is no working image yet, so the original stands in; the compositor makes
        the same choice when it starts a working image from scratch.
        """
        working = (
            self.output_base / INTERMEDIATE_DIR / state.image_name
            / "working" / f"{state.image_name}_working.png"
        )
        return str(working) if working.exists() else state.image_path

    def _save_r1_image(self, image_name: str, image_rgb: np.ndarray) -> str:
        """
        Save R1 gray-masked image.

        Args:
            image_name: Image filename stem
            image_rgb: R1 processed image (RGB)

        Returns:
            Path to saved image
        """
        r1_dir = self.output_base / INTERMEDIATE_DIR / image_name / "round1"
        r1_dir.mkdir(parents=True, exist_ok=True)

        r1_path = r1_dir / f"{image_name}_round1.png"
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(r1_path), image_bgr)

        return str(r1_path)

    # =========================================================================
    # Utility Methods: Regional Crops
    # =========================================================================

    def _calculate_sliding_window_regions(
        self,
        w: int,
        h: int,
        crop_w: int,
        crop_h: int,
        overlap: float
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Calculate all region coordinates using sliding window approach.

        Edge cases:
        - Image smaller than crop: Use full image as single region (no padding)
        - Boundary crops: Include all, even if overlap exceeds specified percentage

        Args:
            w: Image width
            h: Image height
            crop_w: Crop width
            crop_h: Crop height
            overlap: Overlap percentage (0.0 to 1.0)

        Returns:
            List of (region_name, x1, y1, x2, y2) tuples
            Region names are integer strings ("0", "1", "2", ...)
        """
        regions = []

        # Handle image smaller than crop size - skip R2 entirely
        # R2 would just re-evaluate the full image which R1 already did
        if w <= crop_w and h <= crop_h:
            return regions  # Return empty list, no R2 regions needed

        stride_x = int(crop_w * (1 - overlap))
        stride_y = int(crop_h * (1 - overlap))

        # Ensure stride is at least 1 to avoid infinite loop
        stride_x = max(1, stride_x)
        stride_y = max(1, stride_y)

        region_idx = 0
        y = 0
        while y < h:
            x = 0
            while x < w:
                # Calculate crop boundaries
                x1 = x
                y1 = y
                x2 = min(x + crop_w, w)
                y2 = min(y + crop_h, h)

                # Align to boundary if crop would exceed - ensure full crop size
                if x2 == w and x2 - x1 < crop_w:
                    x1 = max(0, w - crop_w)
                if y2 == h and y2 - y1 < crop_h:
                    y1 = max(0, h - crop_h)

                regions.append((str(region_idx), x1, y1, x2, y2))
                region_idx += 1

                # Move to next position or break if at boundary
                if x2 >= w:
                    break
                x += stride_x

            if y2 >= h:
                break
            y += stride_y

        return regions

    def _upscale_region_proportional(
        self,
        region: np.ndarray,
        upscale_target: int
    ) -> np.ndarray:
        """
        Upscale region proportionally based on aspect ratio.

        The longer dimension scales to upscale_target, shorter scales proportionally.
        Example: 768x768 -> 2048x2048, 768x512 -> 2048x1365

        Args:
            region: Region image array (H, W, C)
            upscale_target: Target size for longer dimension

        Returns:
            Upscaled region array
        """
        h, w = region.shape[:2]
        if w >= h:
            new_w = upscale_target
            new_h = int(h * (upscale_target / w))
        else:
            new_h = upscale_target
            new_w = int(w * (upscale_target / h))
        return cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def _extract_and_save_regional_crops(
        self,
        image_name: str,
        image_rgb: np.ndarray,
        iter_idx: int = 1,
    ) -> Dict[str, str]:
        """
        Extract regional crops using sliding window and save for R2 evaluation.

        Uses configurable sliding window parameters from config['round2_regions']:
        - crop_width: Width of each crop
        - crop_height: Height of each crop
        - overlap_percent: Overlap between adjacent crops
        - upscale_target: Target size for longer dimension (proportional scaling)

        Args:
            image_name: Image filename stem
            image_rgb: R1 processed image (RGB)
            iter_idx: Round 2 iteration index — crops are stored under
                intermediate/{name}/r1_crops/iter{iter_idx}/ so each iter's crops
                live separately.

        Returns:
            Dict mapping region names ("0", "1", "2", ...) to crop paths
        """
        h, w = image_rgb.shape[:2]

        # Get sliding window config parameters
        r2_config = self.config.get('round2_regions', {})
        crop_w = r2_config.get('crop_width', 768)
        crop_h = r2_config.get('crop_height', 768)
        overlap = r2_config.get('overlap_percent', 0.1)
        upscale_target = r2_config.get('upscale_target', 2048)

        # Calculate region coordinates using sliding window
        regions = self._calculate_sliding_window_regions(w, h, crop_w, crop_h, overlap)

        # Create crop directory (iter-specific)
        crop_dir = self.output_base / INTERMEDIATE_DIR / image_name / "r1_crops" / f"iter{iter_idx}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        crops = {}
        for region_name, x1, y1, x2, y2 in regions:
            # Extract crop
            crop = image_rgb[y1:y2, x1:x2]

            # Upscale proportionally for GPT evaluation
            upscaled = self._upscale_region_proportional(crop, upscale_target)

            # Save
            crop_path = crop_dir / f"{image_name}_{region_name}.png"
            crop_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)

            crops[region_name] = str(crop_path)

        return crops

    def _extract_region(
        self,
        image_rgb: np.ndarray,
        region: str
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract a specific region from image and upscale.

        Uses sliding window to calculate coordinates based on region index.

        Args:
            image_rgb: Full image (RGB)
            region: Region name (integer string like "0", "1", "2", ...)

        Returns:
            Tuple of (upscaled_region, (x1, y1, x2, y2) crop coords)
        """
        h, w = image_rgb.shape[:2]

        # Get sliding window config parameters
        r2_config = self.config.get('round2_regions', {})
        crop_w = r2_config.get('crop_width', 768)
        crop_h = r2_config.get('crop_height', 768)
        overlap = r2_config.get('overlap_percent', 0.1)
        upscale_target = r2_config.get('upscale_target', 2048)

        # Calculate all regions using sliding window
        regions = self._calculate_sliding_window_regions(w, h, crop_w, crop_h, overlap)

        # Find the requested region by name (index)
        coords = None
        for region_name, x1, y1, x2, y2 in regions:
            if region_name == region:
                coords = (x1, y1, x2, y2)
                break

        if coords is None:
            # Fallback: use first region if requested region not found
            logger.warning(f"[R2 Seg] Region '{region}' not found, using first region")
            coords = (regions[0][1], regions[0][2], regions[0][3], regions[0][4]) if regions else (0, 0, w, h)

        x1, y1, x2, y2 = coords
        crop = image_rgb[y1:y2, x1:x2]

        # Upscale proportionally for segmentation
        upscaled = self._upscale_region_proportional(crop, upscale_target)

        return upscaled, (x1, y1, x2, y2)

    # =========================================================================
    # Utility Methods: SD Queue
    # =========================================================================

    def _push_clues_to_sd_queue(
        self,
        image_id: str,
        image_name: str,
        features: List[Dict],
        masks: List[np.ndarray],
        mask_paths: List[str],
        image_rgb: np.ndarray,
        depth_map: np.ndarray,
        region_name: Optional[str],
        iter_idx: int = 1,
    ):
        """
        Push segmented clues to SD queue for inpainting.

        Args:
            image_id: Image identifier
            image_name: Image filename stem
            features: List of feature dicts
            masks: List of binary masks
            mask_paths: List of mask file paths
            image_rgb: Original image (RGB)
            depth_map: Depth map
            region_name: Region name for R2 (None for R1)
        """
        if self.qm.sd_queue is None:
            logger.warning("[SD Queue] SD queue not initialized")
            return

        round_num = 2 if region_name else 1

        for idx, (feature, mask, mask_path) in enumerate(zip(features, masks, mask_paths)):
            object_desc = feature.get('object_description', '')
            target_desc = feature.get(self.spec.anonymized_field, '')
            geo_desc = feature.get(self.spec.privacy_field, '')
            region_tag = feature.get('region', 'across region')

            # Calculate bbox
            bbox = calculate_mask_bbox(mask)

            # Classify clue
            size_class, crop_size, batch_size = classify_clue(bbox, round=round_num)

            # Create ClueMetadata
            clue_meta = ClueMetadata(
                clue_idx=idx,
                object_description=object_desc,
                target_description=target_desc,
                negative_prompt=geo_desc if geo_desc else "",
                mask=mask,
                bbox=bbox,
                region_tag=region_tag,
                crop_coords=None,
                size_class=size_class,
                crop_size=crop_size,
                batch_size=batch_size,
            )

            # Add to SD queue
            self.qm.sd_queue.add_clue_with_image_id(
                image_id=image_id,
                clue_meta=clue_meta,
                image=image_rgb,
                depth_map=depth_map,
                region_name=region_name,
                mask_path=mask_path,
                iter_idx=iter_idx,
            )

        logger.debug(f"[SD Queue] Added {len(masks)} clues from {image_id} {region_name or 'R1'}")


def start_seg_worker(
    queue_manager: QueueManager,
    global_state: GlobalState,
    sdxl_pipeline,
    config: Dict[str, Any],
    event_loop: asyncio.AbstractEventLoop,
    clue_stats: Optional[ClueStatsTracker] = None
) -> threading.Thread:
    """
    Create and start the segmentation worker thread.

    Args:
        queue_manager: QueueManager instance
        global_state: GlobalState instance
        sdxl_pipeline: SDXLInpaintingPipeline instance
        config: Pipeline configuration
        event_loop: Main asyncio event loop
        clue_stats: Optional ClueStatsTracker for end-of-run accounting

    Returns:
        The started Thread object
    """
    handler = SegmentationQueueHandler(
        queue_manager, global_state, sdxl_pipeline, config, event_loop, clue_stats
    )

    thread = threading.Thread(target=handler.run, daemon=True, name="SegWorker")
    thread.start()

    return thread
