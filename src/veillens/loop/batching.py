"""
Batch Processing Infrastructure for SDXL Inpainting Pipeline

This module provides batch processing capabilities to speed up SDXL inpainting by:
1. Pre-computing the depth map once per image
2. Classifying clues by size and batching similar-sized crops
3. Processing batches through SDXL with shared model state

Expected speedup: 5-15x for images with 5+ clues
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ClueMetadata:
    """Stores clue information and classification results"""
    clue_idx: int
    object_description: str
    target_description: str
    negative_prompt: str
    mask: np.ndarray  # From SAM2 (H, W) uint8
    bbox: Tuple[int, int, int, int]  # From mask (x1, y1, x2, y2)
    region_tag: str  # "across region" or region name
    crop_coords: Optional[Tuple[int, int, int, int]]  # For Round 2 (x1, y1, x2, y2)
    size_class: str  # "small", "large", "full", "regional"
    crop_size: Optional[int]  # Target crop dimension (None for full/regional)
    batch_size: int  # Max batch size


@dataclass
class BatchItem:
    """Single item in a batch queue"""
    clue_idx: int
    image_crop: np.ndarray  # Cropped image (H, W, 3) uint8 RGB
    mask_crop: np.ndarray   # Cropped mask (H, W) uint8
    depth_crop: np.ndarray  # Cropped depth map (H, W) uint8
    positive_prompt: str
    negative_prompt: str
    crop_x1: int  # Full-image x-coordinate
    crop_y1: int  # Full-image y-coordinate
    crop_w: int   # Actual crop width (before padding)
    crop_h: int   # Actual crop height (before padding)
    original_mask: np.ndarray  # Full-size mask for compositing (H, W) uint8
    # Padding information for boundary crops
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0


@dataclass
class BatchItemWithImageId(BatchItem):
    """
    Extended BatchItem with cross-image tracking.

    This dataclass extends BatchItem to support cross-image batching by tracking
    which image each clue belongs to, and for Round 2, which region the clue came from.

    Attributes:
        image_id: Unique identifier for the source image
        region_name: Region name for Round 2 (e.g., "top_left", "middle"), None for Round 1
        mask_path: Path to the mask file for phase_cache tracking
        iter_idx: Iteration index (0 = the R1-only iteration, then 1..max_iter)
    """
    image_id: str = ""
    region_name: Optional[str] = None
    mask_path: str = ""
    iter_idx: int = 1


def calculate_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box from SAM2 mask.

    Args:
        mask: Binary mask array (H, W) with values > 0 indicating object

    Returns:
        Tuple (x1, y1, x2, y2) bounding box coordinates
        Returns (0, 0, 0, 0) if mask is empty
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, 0, 0)  # Empty mask

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    # Add 1 to x2, y2 to make exclusive (standard bbox format)
    return (int(x1), int(y1), int(x2) + 1, int(y2) + 1)


#: Size class -> (crop_size, batch_size). ``crop_size`` is None for the full-image class.
#: A clue is assigned a class by its longest bounding-box side; the thresholds are 0.7x the
#: crop size of the class below, so a clue always fits inside the crop it is given.
SIZE_CLASSES = {
    "small":  (512, 12),
    "median": (768, 8),
    "large":  (1024, 4),
    "full":   (None, 1),
}


def classify_clue(bbox: Tuple[int, int, int, int], round: int) -> Tuple[str, Optional[int], int]:
    """
    Classify clue based on bounding box dimensions using consistent 4-class system.

    Both Round 1 and Round 2 use the same classification to ensure dimension consistency:
    - small: max_dim <= 252 (0.7 * 360) → 512x512 crop, batch_size 12
    - median: max_dim <= 358.4 (0.7 * 512) → 768x768 crop, batch_size 8
    - large: max_dim <= 537.6 (0.7 * 768) → 1024x1024 crop, batch_size 4
    - full: max_dim > 537.6 → full image (1920x1080), batch_size 1

    Args:
        bbox: (x1, y1, x2, y2) bounding box coordinates in full image space
        round: 1 for full image, 2 for regional crops (used for context only)

    Returns:
        Tuple of (class_name, crop_size, batch_size)
        crop_size is None for "full" class
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    max_dim = max(width, height)

    # Consistent 4-class system for both rounds
    if max_dim <= 252:  # 0.7 * 360
        name = "small"
    elif max_dim <= 358.4:  # 0.7 * 512
        name = "median"
    elif max_dim <= 537.6:  # 0.7 * 768
        name = "large"
    else:
        name = "full"
    return (name, *SIZE_CLASSES[name])


def extract_clue_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    crop_size: int,
    depth_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Extract square crop around clue center with edge padding if needed.

    Args:
        image: Full image (H, W, 3) RGB uint8
        mask: Full mask (H, W) uint8
        bbox: Bounding box (x1, y1, x2, y2)
        crop_size: Target crop dimension (square)
        depth_map: Full depth map (H, W) uint8

    Returns:
        Tuple of (image_crop, mask_crop, depth_crop, crop_coords, padding)
        - All crops are crop_size × crop_size (padded if necessary)
        - crop_coords: (crop_x1, crop_y1, crop_w, crop_h) - actual region extracted (before padding)
        - padding: (pad_top, pad_bottom, pad_left, pad_right) - padding applied to crops
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Calculate center of bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Calculate crop boundaries (square crop)
    half_size = crop_size // 2
    crop_x1 = center_x - half_size
    crop_y1 = center_y - half_size
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    # Calculate padding needed (if crop extends beyond image boundaries)
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)

    # Adjust crop coordinates to stay within image
    crop_x1_adj = max(0, crop_x1)
    crop_y1_adj = max(0, crop_y1)
    crop_x2_adj = min(w, crop_x2)
    crop_y2_adj = min(h, crop_y2)

    # Extract crops
    image_crop = image[crop_y1_adj:crop_y2_adj, crop_x1_adj:crop_x2_adj]
    mask_crop = mask[crop_y1_adj:crop_y2_adj, crop_x1_adj:crop_x2_adj]
    depth_crop = depth_map[crop_y1_adj:crop_y2_adj, crop_x1_adj:crop_x2_adj]

    # Apply edge padding if needed (reflection padding for images, constant for mask)
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        # BORDER_REFLECT_101: Mirrors pixels without repeating edges
        image_crop = cv2.copyMakeBorder(
            image_crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101
        )
        # Mask: pad with zeros (no masking outside clue region)
        mask_crop = cv2.copyMakeBorder(
            mask_crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        depth_crop = cv2.copyMakeBorder(
            depth_crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101
        )

    # Track actual crop coordinates in full image (before padding)
    crop_coords = (
        crop_x1_adj,
        crop_y1_adj,
        crop_x2_adj - crop_x1_adj,  # width
        crop_y2_adj - crop_y1_adj   # height
    )

    # Track padding applied to crops
    padding = (pad_top, pad_bottom, pad_left, pad_right)

    return image_crop, mask_crop, depth_crop, crop_coords, padding


class BatchQueueManager:
    """Manages batch queues for different size classes"""

    def __init__(self, round: int, config: Dict):
        """
        Initialize batch queue manager.

        Args:
            round: 1 for full image, 2 for regional crops
            config: Pipeline configuration dict
        """
        self.round = round
        self.config = config

        # One queue per size class, sharing the table classify_clue assigns from
        self.queues = {
            name: {"crop_size": crop_size, "batch_size": batch_size, "items": []}
            for name, (crop_size, batch_size) in SIZE_CLASSES.items()
        }

    def get_queue_summary(self) -> str:
        """Get summary of items in queues for logging"""
        summary_parts = []
        for size_class, queue_info in self.queues.items():
            num_items = len(queue_info["items"])
            if num_items > 0:
                summary_parts.append(f"{size_class}:{num_items}")

        return ", ".join(summary_parts) if summary_parts else "empty"


class GlobalBatchQueueManager(BatchQueueManager):
    """
    Extended BatchQueueManager for cross-image batching.

    This class extends BatchQueueManager to support batching clues from multiple
    images. It implements a hybrid flush strategy that flushes batches when either:
    - Batch size is reached (size-based flush)
    - All images have been segmented and remaining clues are queued (completion-based flush)

    Attributes:
        total_images_to_process: Total number of images to process in this batch
        images_segmented: Number of images that have completed segmentation
    """

    def __init__(self, round: int, config: Dict, flush_strategy: str = "hybrid"):
        """
        Initialize global batch queue manager for cross-image batching.

        Args:
            round: 1 for full image, 2 for regional crops
            config: Pipeline configuration dict
            flush_strategy: Deprecated parameter, always uses "hybrid" strategy
        """
        super().__init__(round, config)
        self.total_images_to_process = 0
        self.images_segmented = 0

    def add_clue_with_image_id(
        self,
        image_id: str,
        clue_meta: ClueMetadata,
        image: np.ndarray,
        depth_map: np.ndarray,
        region_name: Optional[str] = None,
        mask_path: str = "",
        iter_idx: int = 1,
    ):
        """
        Add clue to appropriate queue with image_id tracking.

        This method extends the base add_clue() to create BatchItemWithImageId
        objects that track which image each clue belongs to, enabling cross-image
        batching and immediate compositing.

        Args:
            image_id: Unique identifier for the source image
            clue_meta: Clue metadata with classification
            image: Full image (H, W, 3) RGB uint8
            depth_map: Pre-computed depth map (full image) (H, W) uint8
            region_name: Region name for Round 2 (e.g., "top_left"), None for Round 1
            mask_path: Path to the mask file for phase_cache tracking
        """
        size_class = clue_meta.size_class

        if size_class == "full":
            # No cropping, use full image
            image_crop = image
            mask_crop = clue_meta.mask
            depth_crop = depth_map
            crop_coords = (0, 0, image.shape[1], image.shape[0])
            padding = (0, 0, 0, 0)  # No padding for full image

        else:
            # Extract crop around clue center
            crop_size = clue_meta.crop_size
            image_crop, mask_crop, depth_crop, crop_coords, padding = extract_clue_crop(
                image, clue_meta.mask, clue_meta.bbox, crop_size, depth_map
            )

        # Create BatchItemWithImageId (extends BatchItem with image_id tracking)
        batch_item = BatchItemWithImageId(
            clue_idx=clue_meta.clue_idx,
            image_crop=image_crop,
            mask_crop=mask_crop,
            depth_crop=depth_crop,
            positive_prompt=clue_meta.target_description,
            negative_prompt=clue_meta.negative_prompt,
            crop_x1=crop_coords[0],
            crop_y1=crop_coords[1],
            crop_w=crop_coords[2],
            crop_h=crop_coords[3],
            original_mask=clue_meta.mask,
            pad_top=padding[0],
            pad_bottom=padding[1],
            pad_left=padding[2],
            pad_right=padding[3],
            image_id=image_id,
            region_name=region_name,
            mask_path=mask_path,
            iter_idx=iter_idx,
        )

        self.queues[size_class]["items"].append(batch_item)

    def should_flush(self, queue_name: str,
                     seg_empty_duration: float = 0.0, timeout: float = 5.0) -> bool:
        """
        Decide whether a size-class queue should be handed to SDXL.

        The single rule is GPU exclusivity: diffusion starts only once the segmentation
        queue has been idle for `timeout` seconds, so SAM 2 and SDXL never contend for
        the device. A full batch does NOT trigger a flush on its own -- batching is an
        efficiency, exclusivity is a correctness constraint.

        Args:
            queue_name: Name of the queue to check (e.g., "small", "large", "full")
            seg_empty_duration: How long (seconds) the seg queue has been empty
            timeout: Seconds of segmentation idleness required before flushing

        Returns:
            True if the queue should be flushed, False otherwise
        """
        if not self.queues[queue_name]["items"]:
            return False
        return seg_empty_duration >= timeout

    def get_ready_batches(self, seg_empty_duration: float = 0.0,
                          timeout: float = 5.0) -> List[Tuple[str, List[BatchItemWithImageId]]]:
        """
        Get batches that are ready to flush based on the flush strategy.

        This method checks all queues and returns batches that should be flushed
        according to the configured flush strategy.

        Args:
            seg_empty_duration: How long (seconds) the seg queue has been empty
            timeout: Seconds of segmentation idleness required before flushing

        Returns:
            List of (size_class, batch_items) tuples for batches ready to process
        """
        ready_batches = []

        for size_class, queue_info in self.queues.items():
            items = queue_info["items"]
            batch_size = queue_info["batch_size"]

            if not self.should_flush(size_class, seg_empty_duration, timeout):
                continue

            # Pop a batch from the queue
            batch = items[:batch_size]
            self.queues[size_class]["items"] = items[batch_size:]

            ready_batches.append((size_class, batch))

        return ready_batches

    def get_total_items(self) -> int:
        """
        Get total number of items across all queues.

        Returns:
            Total item count
        """
        return sum(len(q["items"]) for q in self.queues.values())

    def get_all_batches(self) -> List[Tuple[str, List[BatchItemWithImageId]]]:
        """
        Get all batches from all queues (override parent to handle BatchItemWithImageId).

        Returns:
            List of (size_class, batch) tuples
            Each batch is a list of BatchItemWithImageId objects
        """
        all_batches = []

        for size_class, queue_info in self.queues.items():
            items = queue_info["items"]
            batch_size = queue_info["batch_size"]

            if not items:
                continue

            # Split items into batches of max batch_size
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                all_batches.append((size_class, batch))

            # Clear the queue after getting all batches
            self.queues[size_class]["items"] = []

        return all_batches
