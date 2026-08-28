"""
Global State Management for Parallel Pipeline

This module provides thread-safe global state tracking for the parallel
geolocation clue removal pipeline. It maintains per-image processing state
and a global completion counter.
"""

from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Dict, List, Optional, Tuple

import numpy as np

# Geolocation mask levels — must match prompt/geo/schema_original.json:properties.enum
LEVELS: Tuple[str, ...] = (
    "country level clue",
    "city level clue",
    "street level clue",
)
# Order from most to least specific. A feature tagged with multiple levels
# is counted under the most specific one so totals reflect distinct clues.
LEVELS_BY_SPECIFICITY: Tuple[str, ...] = (
    "street level clue",
    "city level clue",
    "country level clue",
)
ROUND_TAGS: Tuple[str, ...] = ("R1", "R2")


@dataclass
class ImageState:
    """
    Per-image processing state tracker.

    Tracks the complete lifecycle of an image through the pipeline:
    - R1 evaluation and segmentation
    - R2 regional evaluation and segmentation
    - SD inpainting progress
    - Final evaluation

    Attributes:
        image_id: Unique identifier for the image
        image_path: Path to original input image
        image_name: Stem of image filename (without extension)
        init_pred_loc: Predicted location from R1 evaluation (for R2 context)
        ground_truth: Ground truth coordinates from dataset
        r1_pred_path: Path to cached R1 prediction JSON
        depth_path: Path to cached depth map
        r1_final_image: Path to gray-masked R1 image (for R2 evaluation)
        r2_pred_paths: List of paths to R2 regional prediction JSONs
        r2_regions_received: Count of R2 predictions received
        expected_r2_regions: Number of R2 regions expected (set during R1 completion)
        seg_completed: Whether all segmentation is complete
        r2_seg_complete: Whether R2 segmentation specifically is complete
        r2_seg_count: Number of R2 regions segmented
        final_image: Path to final processed image
        final_eval: Path to final evaluation JSON
        num_total_clue: Total clues to process (-1 = not yet determined)
        num_finished_clue: Number of clues with SD inpainting complete
        working_image: In-memory working image for compositing (numpy array)
        mask_paths: List of all mask paths for phase_cache tracking
    """
    image_id: str
    image_path: str
    image_name: str = ""

    # Location tracking
    init_pred_loc: Optional[str] = None
    ground_truth: Optional[Dict[str, float]] = None

    # R1 state
    r1_pred_path: Optional[str] = None
    depth_path: Optional[str] = None
    r1_final_image: Optional[str] = None

    # R2 state
    r2_pred_paths: List[str] = field(default_factory=list)
    r2_regions_received: int = 0
    expected_r2_regions: int = 0  # Number of R2 regions expected (set during R1 completion)
    r2_seg_complete: bool = False
    r2_seg_count: int = 0

    # Multi-iteration R2 tracking
    # r2_iter: 0 = the R1-only iteration, 1 = first R2 iteration, 2 = second, ...
    r2_iter: int = 0
    # Set by gate eval when err_km >= val_dist; honored by compositor on next is_complete()
    stop_after_current_iter: bool = False

    # Completion flags
    seg_completed: bool = False
    final_image: Optional[str] = None
    final_eval: Optional[str] = None

    # Clue tracking
    num_total_clue: int = -1  # -1 = not yet set, distinguishes from 0 clues
    num_finished_clue: int = 0

    # In-memory state (not persisted)
    working_image: Optional[np.ndarray] = field(default=None, repr=False)
    mask_paths: List[str] = field(default_factory=list)

    # Thread safety for this image's state
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self):
        """Extract image_name from image_path if not provided."""
        if not self.image_name and self.image_path:
            from pathlib import Path
            self.image_name = Path(self.image_path).stem

    def add_r2_pred_path(self, path: str) -> int:
        """
        Thread-safe addition of R2 prediction path.

        Returns:
            Number of R2 predictions received after addition
        """
        with self._lock:
            self.r2_pred_paths.append(path)
            self.r2_regions_received = len(self.r2_pred_paths)
            return self.r2_regions_received

    def increment_r2_seg_count(self) -> int:
        """
        Thread-safe increment of R2 segmentation count.

        Returns:
            Count after increment. Sets r2_seg_complete=True when count reaches expected_r2_regions.
        """
        with self._lock:
            self.r2_seg_count += 1
            if self.expected_r2_regions > 0 and self.r2_seg_count >= self.expected_r2_regions:
                self.r2_seg_complete = True
            return self.r2_seg_count

    def set_expected_r2_regions(self, count: int) -> None:
        """
        Thread-safe setter for expected R2 region count.

        When count is 0 (image smaller than crop), immediately marks
        R2 segmentation as complete since there's nothing to wait for.

        Args:
            count: Number of R2 regions expected (based on sliding window)
        """
        with self._lock:
            self.expected_r2_regions = count
            # If no R2 regions expected, mark R2 seg as complete immediately
            if count == 0:
                self.r2_seg_complete = True

    def increment_finished_clue(self) -> int:
        """
        Thread-safe increment of finished clue count.

        Returns:
            Count after increment
        """
        with self._lock:
            self.num_finished_clue += 1
            return self.num_finished_clue

    def set_num_total_clue(self, count: int) -> None:
        """Thread-safe setter for total clue count (R1 sets initial value)."""
        with self._lock:
            if self.num_total_clue == -1:
                self.num_total_clue = count
            else:
                # R2 adds to existing count
                self.num_total_clue += count

    def add_to_total_clue(self, count: int) -> int:
        """
        Thread-safe addition to total clue count (for R2 regions).

        Returns:
            New total after addition
        """
        with self._lock:
            if self.num_total_clue == -1:
                self.num_total_clue = count
            else:
                self.num_total_clue += count
            return self.num_total_clue

    def is_complete(self) -> bool:
        """
        Check if image processing is complete.

        Completion requires:
        - R2 segmentation complete (auto-set to True when expected_r2_regions == 0)
        - All R2 predictions received (0 == 0 is valid for small images)
        - All clues have been inpainted (num_finished == num_total)
        - num_total_clue is set (>= 0)

        Note: When expected_r2_regions == 0 (image smaller than crop),
        r2_seg_complete is set to True immediately in set_expected_r2_regions().
        """
        with self._lock:
            # Use r2_seg_complete as the gate instead of expected_r2_regions > 0
            # r2_seg_complete is True when:
            # - 0 regions expected (set immediately in set_expected_r2_regions)
            # - All N regions have been segmented
            return (
                self.r2_seg_complete and
                len(self.r2_pred_paths) == self.expected_r2_regions and
                self.num_total_clue >= 0 and
                self.num_finished_clue >= self.num_total_clue
            )

    def add_mask_path(self, path: str) -> None:
        """Thread-safe addition of mask path for tracking."""
        with self._lock:
            self.mask_paths.append(path)

    def reset_for_next_iter(self) -> None:
        """
        Reset per-iteration R2 counters before starting iter (i+1).

        Preserves: working_image, mask_paths (cumulative across iters),
        r1_final_image, depth path, init_pred_loc (will be refreshed by gate).

        Resets: r2_pred_paths, expected_r2_regions, r2_seg_count, r2_seg_complete,
        and the clue budget (num_total_clue, num_finished_clue) so that is_complete()
        fires when THIS iter's R2 clues are all processed.
        """
        with self._lock:
            self.r2_pred_paths = []
            self.r2_regions_received = 0
            self.expected_r2_regions = 0
            self.r2_seg_complete = False
            self.r2_seg_count = 0
            self.num_total_clue = 0
            self.num_finished_clue = 0


class GlobalState:
    """
    Thread-safe global state manager for the parallel pipeline.

    Manages:
    - Per-image ImageState instances
    - Global completion counter
    - Total images to process

    Thread Safety:
    - Uses RLock for nested locking scenarios
    - All mutations go through locked methods
    """

    def __init__(self):
        self.images: Dict[str, ImageState] = {}
        self.num_complete: int = 0
        self.total_images: int = 0
        self._lock = RLock()

    def add_image(self, image_id: str, image_path: str,
                  ground_truth: Optional[Dict[str, float]] = None) -> ImageState:
        """
        Add a new image to track.

        Args:
            image_id: Unique identifier
            image_path: Path to image file
            ground_truth: Optional ground truth coordinates

        Returns:
            The created ImageState instance
        """
        with self._lock:
            state = ImageState(
                image_id=image_id,
                image_path=image_path,
                ground_truth=ground_truth
            )
            self.images[image_id] = state
            return state

    def get_image(self, image_id: str) -> Optional[ImageState]:
        """Get ImageState for an image (thread-safe read)."""
        with self._lock:
            return self.images.get(image_id)

    def increment_complete(self) -> int:
        """
        Atomically increment completion counter.

        Returns:
            New completion count
        """
        with self._lock:
            self.num_complete += 1
            return self.num_complete

    def set_total_images(self, count: int) -> None:
        """Set total number of images to process."""
        with self._lock:
            self.total_images = count

    def is_all_complete(self) -> bool:
        """Check if all images are complete."""
        with self._lock:
            return self.num_complete >= self.total_images and self.total_images > 0

    def get_progress(self) -> tuple:
        """
        Get current progress.

        Returns:
            Tuple of (completed, total)
        """
        with self._lock:
            return (self.num_complete, self.total_images)

    def get_pending_images(self) -> List[str]:
        """Get list of image IDs that are not yet complete."""
        with self._lock:
            return [
                image_id for image_id, state in self.images.items()
                if state.final_eval is None
            ]


@dataclass
class ClueStatsTracker:
    """
    Aggregate counters for clues attempted/skipped during segmentation.

    Buckets are keyed by (round_tag, level) where round_tag ∈ ROUND_TAGS and
    level ∈ LEVELS. Each clue is counted exactly once, under its most
    specific level (street > city > country), so per-round totals equal the
    number of distinct clues and a success rate can be derived directly.
    Features whose `properties` field has no recognized level are silently
    ignored — this keeps the tracker passive for the personal-info spec.
    """
    selected: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: {(r, l): 0 for r in ROUND_TAGS for l in LEVELS}
    )
    skipped: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: {(r, l): 0 for r in ROUND_TAGS for l in LEVELS}
    )
    _lock: Lock = field(default_factory=Lock, repr=False)

    @staticmethod
    def _most_specific_level(feature: Dict) -> Optional[str]:
        levels = feature.get("properties") or []
        for level in LEVELS_BY_SPECIFICITY:
            if level in levels:
                return level
        return None

    def _bump(self, bucket: Dict[Tuple[str, str], int], round_tag: str, feature: Dict) -> None:
        level = self._most_specific_level(feature)
        if level is None:
            return
        key = (round_tag, level)
        with self._lock:
            if key in bucket:
                bucket[key] += 1

    def record_selected(self, round_tag: str, feature: Dict) -> None:
        """Record that a clue passed mask_level filtering and we attempted to ground it."""
        self._bump(self.selected, round_tag, feature)

    def record_skipped(self, round_tag: str, feature: Dict) -> None:
        """Record that a clue was skipped because Grounded SAM 2 failed to ground it."""
        self._bump(self.skipped, round_tag, feature)

    def has_data(self) -> bool:
        with self._lock:
            return any(self.selected.values()) or any(self.skipped.values())

    def snapshot(self) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
        with self._lock:
            return dict(self.selected), dict(self.skipped)

    def assert_something_was_grounded(self) -> None:
        """
        Raise unless at least one selected clue was actually localised.

        If the oracle proposed clues and not one of them could be grounded, no image was edited,
        and the "protected" outputs are byte-for-byte the originals. The evaluation still runs
        and still produces a table -- the attacker simply re-reads the same photographs -- so a
        broken install is indistinguishable from a weak result. That is what an unbuilt
        GroundingDINO extension does: every detection raises `name '_C' is not defined`, the loop
        logs "no ground" per clue, and the run completes with exit 0.

        Zero clues selected is a legitimate outcome (the oracle found nothing to remove). Zero
        grounded out of many selected is not.
        """
        if not self.has_data():
            return
        selected, skipped = self.snapshot()
        n_selected = sum(selected.values())
        n_grounded = n_selected - sum(skipped.values())
        if n_selected == 0 or n_grounded > 0:
            return
        raise RuntimeError(
            f"grounding produced nothing: {n_selected} clues were selected and 0 were localised, "
            "so no image was edited and the evaluation compares the originals with themselves.\n"
            "The usual cause is GroundingDINO's compiled extension being absent, which makes "
            "every detection raise `name '_C' is not defined`; the per-clue errors are in the log "
            "at ERROR level. Run `veillens-protect --config <config> --dry-run`, which checks "
            "for it."
        )
