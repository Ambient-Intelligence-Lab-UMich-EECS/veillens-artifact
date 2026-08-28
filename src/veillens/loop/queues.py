"""
Queue Manager for Parallel Pipeline

This module provides the three-queue system for complete parallelism:
- api_queue: Handles all GPT API requests (R1 eval, R2 eval, final eval)
- seg_queue: Handles GPU segmentation tasks
- sd_queue: Handles batched Stable Diffusion inpainting (reuses existing GlobalBatchQueueManager)

The queues operate independently, enabling API requests to proceed while
GPU is busy with segmentation or inpainting.
"""

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class TaskType(Enum):
    """Types of API tasks."""
    R1_EVAL = "r1_eval"       # Round 1 full image evaluation
    R2_EVAL = "r2_eval"       # Round 2 regional evaluation
    GATE_EVAL = "gate_eval"   # Per-iteration full-image gate evaluation
    FINAL_EVAL = "final_eval"  # Final processed image evaluation


@dataclass
class APITask:
    """
    Task for the API queue.

    Attributes:
        image_id: Unique identifier for the image
        image_path: Path to image file to evaluate
        task_type: Type of evaluation task
        region: Region name for R2 tasks (top_left, top_right, etc.)
        iter_idx: Iteration index (0 = the R1-only iteration, then 1..max_iter)
    """
    image_id: str
    image_path: str
    task_type: TaskType
    region: Optional[str] = None
    iter_idx: int = 1

    def __repr__(self):
        if self.region:
            return f"APITask({self.image_id}, {self.task_type.value}, {self.region}, iter={self.iter_idx})"
        return f"APITask({self.image_id}, {self.task_type.value}, iter={self.iter_idx})"


@dataclass
class SegTask:
    """
    Task for the segmentation queue.

    Attributes:
        image_id: Unique identifier for the image
        region: Region being processed (full_img, r2_setup, "0"/"1"/.., etc.)
        image_path: Path to image file for segmentation
        pred_path: Path to prediction JSON with features to segment
        iter_idx: Iteration index (0 = the R1-only iteration, then 1..max_iter)
    """
    image_id: str
    region: str  # full_img, r2_setup, "0", "1", ...
    image_path: str
    pred_path: str
    iter_idx: int = 1

    def __repr__(self):
        return f"SegTask({self.image_id}, {self.region}, iter={self.iter_idx})"


class QueueManager:
    """
    Manages the three-queue system for parallel pipeline execution.

    Queue Descriptions:
    - api_queue (asyncio.Queue): Async queue for GPT API requests
    - seg_queue (queue.Queue): Thread-safe queue for GPU segmentation
    - sd_queue: GlobalBatchQueueManager (initialized separately)

    The manager also tracks timing for SD queue flush decisions.

    Attributes:
        api_queue: Async queue for API tasks
        seg_queue: Thread-safe queue for segmentation tasks
        sd_queue: Reference to GlobalBatchQueueManager (set externally)
        sd_flush_timeout: Seconds of segmentation idleness required before SDXL may run
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize queue manager.

        Args:
            config: Pipeline configuration dict
        """
        self.config = config

        # API queue (asyncio for async workers)
        self.api_queue: asyncio.Queue = asyncio.Queue()

        # Segmentation queue (thread-safe for GPU worker thread)
        self.seg_queue: queue.Queue = queue.Queue()

        # SD queue - will be set to existing GlobalBatchQueueManager
        self.sd_queue = None

        # Flush timing
        self.sd_flush_timeout = 3.0
        self._seg_queue_empty_since: Optional[float] = None

        # Shutdown flag
        self._shutdown = False

        # SD running flag (for GPU coordination)
        self._sd_running = False
        self._sd_running_lock = threading.Lock()

    def set_sd_queue(self, sd_queue) -> None:
        """
        Set the SD queue (GlobalBatchQueueManager instance).

        Args:
            sd_queue: GlobalBatchQueueManager instance
        """
        self.sd_queue = sd_queue

    def set_sd_running(self, running: bool) -> None:
        """
        Set whether SD is currently running (for GPU coordination).

        Thread-safe setter to coordinate GPU usage between seg worker and SD.

        Args:
            running: True if SD is processing a batch, False otherwise
        """
        with self._sd_running_lock:
            self._sd_running = running

    def is_sd_running(self) -> bool:
        """
        Check if SD is currently running.

        Thread-safe getter for seg worker to check before processing tasks.

        Returns:
            True if SD is processing a batch, False otherwise
        """
        with self._sd_running_lock:
            return self._sd_running

    # =========================================================================
    # API Queue Methods
    # =========================================================================

    async def put_api_task(self, task: APITask) -> None:
        """Add task to API queue (async)."""
        await self.api_queue.put(task)

    def put_api_task_nowait(self, task: APITask) -> None:
        """Add task to API queue without waiting (from sync context)."""
        self.api_queue.put_nowait(task)

    async def get_api_task(self) -> APITask:
        """Get next task from API queue (async, blocks until available)."""
        return await self.api_queue.get()

    def api_task_done(self) -> None:
        """Mark current API task as complete."""
        self.api_queue.task_done()

    def api_queue_size(self) -> int:
        """Get number of tasks in API queue."""
        return self.api_queue.qsize()

    # =========================================================================
    # Segmentation Queue Methods
    # =========================================================================

    def put_seg_task(self, task: SegTask) -> None:
        """Add task to segmentation queue (thread-safe)."""
        self.seg_queue.put(task)

    def get_seg_task(self, timeout: Optional[float] = None) -> Optional[SegTask]:
        """
        Get next task from segmentation queue (blocking).

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            SegTask or None if timeout/shutdown
        """
        try:
            return self.seg_queue.get(timeout=timeout or 1.0)
        except queue.Empty:
            return None

    def seg_task_done(self) -> None:
        """Mark current segmentation task as complete."""
        self.seg_queue.task_done()

    def seg_queue_empty(self) -> bool:
        """Check if segmentation queue is empty."""
        return self.seg_queue.empty()

    def seg_queue_size(self) -> int:
        """Get number of tasks in segmentation queue."""
        return self.seg_queue.qsize()

    # =========================================================================
    # SD Queue Flush Logic
    # =========================================================================

    def should_flush_sd(self) -> bool:
        """
        Determine if SD queue should be flushed.

        Flush conditions:
        1. Batch size reached for any queue (handled by GlobalBatchQueueManager)
        2. Seg queue has been empty for >= sd_flush_timeout seconds AND sd_queue has items

        Returns:
            True if SD queue should be flushed
        """
        current_time = time.time()

        if self.seg_queue.empty():
            # Start tracking empty time
            if self._seg_queue_empty_since is None:
                self._seg_queue_empty_since = current_time
            elif current_time - self._seg_queue_empty_since >= self.sd_flush_timeout:
                # Timeout reached - check if SD queue has items
                if self.sd_queue and self.sd_queue.get_total_items() > 0:
                    return True
        else:
            # Reset empty tracking
            self._seg_queue_empty_since = None

        return False

    def get_seg_empty_duration(self) -> float:
        """
        Get how long seg_queue has been empty.

        Returns:
            Seconds since seg_queue became empty, or 0 if not empty
        """
        if self._seg_queue_empty_since is None:
            return 0.0
        return time.time() - self._seg_queue_empty_since

    def reset_seg_empty_timer(self) -> None:
        """Reset the seg queue empty timer (after SD flush)."""
        self._seg_queue_empty_since = None

    # =========================================================================
    # Queue Status
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of all queues.

        Returns:
            Dict with queue sizes and timing info
        """
        status = {
            'api_queue_size': self.api_queue.qsize(),
            'seg_queue_size': self.seg_queue.qsize(),
            'seg_empty_duration': self.get_seg_empty_duration(),
            'sd_flush_pending': self.should_flush_sd()
        }

        if self.sd_queue:
            status['sd_queue_summary'] = self.sd_queue.get_queue_summary()
            status['sd_queue_total'] = self.sd_queue.get_total_items()

        return status

    # =========================================================================
    # Shutdown
    # =========================================================================

    def shutdown(self) -> None:
        """Signal shutdown to all queue workers."""
        self._shutdown = True

    def is_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown


# Convenience functions for creating tasks

def create_r1_eval_task(image_id: str, image_path: str) -> APITask:
    """Create a Round 1 evaluation task."""
    return APITask(
        image_id=image_id,
        image_path=image_path,
        task_type=TaskType.R1_EVAL
    )


def create_r2_eval_task(image_id: str, image_path: str, region: str, iter_idx: int = 1) -> APITask:
    """Create a Round 2 regional evaluation task."""
    return APITask(
        image_id=image_id,
        image_path=image_path,
        task_type=TaskType.R2_EVAL,
        region=region,
        iter_idx=iter_idx,
    )


def create_gate_eval_task(image_id: str, image_path: str, iter_idx: int) -> APITask:
    """Create a per-iteration full-image gate evaluation task."""
    return APITask(
        image_id=image_id,
        image_path=image_path,
        task_type=TaskType.GATE_EVAL,
        iter_idx=iter_idx,
    )


def create_final_eval_task(image_id: str, image_path: str) -> APITask:
    """Create a final evaluation task."""
    return APITask(
        image_id=image_id,
        image_path=image_path,
        task_type=TaskType.FINAL_EVAL
    )


def create_seg_task(image_id: str, region: str, image_path: str, pred_path: str, iter_idx: int = 1) -> SegTask:
    """Create a segmentation task."""
    return SegTask(
        image_id=image_id,
        region=region,
        image_path=image_path,
        pred_path=pred_path,
        iter_idx=iter_idx,
    )
