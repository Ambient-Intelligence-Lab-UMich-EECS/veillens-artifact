"""
Dataset-supplied subject boxes: the overlay that names them, and the face box derived from one.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

__all__ = ["BBOX_COLORS", "head_band", "draw_subject_boxes"]

#: Box colours, cycled by subject index. Twelve distinct hues, so an image with more people than
#: colours repeats rather than fading into near-identical shades.
BBOX_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 215, 0),
    (255, 0, 255), (0, 255, 255), (255, 140, 0), (138, 43, 226),
    (0, 255, 127), (220, 20, 60), (30, 144, 255), (255, 20, 147),
)


def head_band(person_mask: np.ndarray,
              head_fraction: float = 0.30,
              width_fraction: float = 1.0,
              head_width_multiple: float = 1.6) -> np.ndarray:
    """
    Narrow a person mask to the head region at its top.

    The band is intersected back with the mask itself, so only pixels that were part of the person
    survive and it can never spill onto the scene behind them -- which is why no horizontal
    clipping is needed by default.

    Its height is the larger of two estimates, because each fails on the pose the other handles:

    * ``head_width_multiple`` x the mask's width in its topmost rows. Those rows are the head, so
      their width is a head width, and a head is a little taller than it is wide whatever the
      subject is doing. Pose-invariant, and the rule that carries crouched and bent-forward
      subjects.
    * ``head_fraction`` x the mask's total height. Sound for an upright figure, and kept as a
      floor so a head seen edge-on -- narrow, so a small width estimate -- is still covered.

    Taking the larger means both rules can only extend the band. For an edit whose job is to
    remove something, over-covering costs image quality and under-covering costs the whole point.

    *width_fraction* below 1.0 clips the band horizontally, centred on the columns the head rows
    occupy rather than on the band as a whole: a bent-forward head sits off its body's centreline,
    so centring on the full band would pull the kept range toward the torso and clip the face.

    Args:
        person_mask: ``(H, W)``, non-zero where the person is. Any dtype.
        head_fraction: floor on the band height, as a share of the mask's total height.
        width_fraction: share of the head rows' width, centred, to keep. 1.0 keeps the band as-is.
        head_width_multiple: band height as a multiple of the head's width.

    Returns:
        ``(H, W)`` bool array. Empty in, empty out -- the caller already treats an empty mask as
        "this clue could not be grounded", which is the right reading here too.
    """
    binary = np.asarray(person_mask) > 0
    rows = np.flatnonzero(binary.any(axis=1))
    if rows.size == 0:
        return np.zeros(binary.shape, dtype=bool)

    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1

    # The head's width, read off the topmost slice of the silhouette. A tenth of the figure is
    # shallow enough to be scalp on any pose, and at least two rows so a small mask still has one.
    probe = binary[top:min(top + max(height // 10, 2), binary.shape[0]), :]
    probe_columns = np.flatnonzero(probe.any(axis=0))
    head_width = int(probe_columns[-1] - probe_columns[0] + 1) if probe_columns.size else 0

    # At least one row: a mask a few pixels tall is a distant face, not a reason to return nothing.
    band_height = max(
        int(round(head_width * float(head_width_multiple))),
        int(round(height * float(head_fraction))),
        1,
    )
    band_bottom = min(top + band_height, binary.shape[0])

    band = np.zeros(binary.shape, dtype=bool)
    band[top:band_bottom, :] = binary[top:band_bottom, :]

    if width_fraction >= 1.0:
        return band

    # Measured on the head rows alone; see the docstring.
    head_rows = band[top:min(top + max(band_height // 3, 1), band.shape[0]), :]
    columns = np.flatnonzero(head_rows.any(axis=0))
    if columns.size == 0:
        return band

    left, right = int(columns[0]), int(columns[-1])
    span = right - left + 1
    keep = max(int(round(span * float(width_fraction))), 1)
    centre = (left + right) / 2.0
    keep_left = max(int(round(centre - keep / 2.0)), 0)
    keep_right = min(keep_left + keep, binary.shape[1])

    band[:, :keep_left] = False
    band[:, keep_right:] = False
    return band


def draw_subject_boxes(image_rgb: np.ndarray,
                       boxes: Sequence[Sequence[float]],
                       ids: Iterable[object] | None = None,
                       *, thickness: int = 3) -> np.ndarray:
    """
    Draw one labelled rectangle per subject onto a copy of *image_rgb*.

    Args:
        image_rgb: ``(H, W, 3)`` uint8 RGB. Not modified.
        boxes: one ``(x1, y1, x2, y2)`` per subject, in absolute pixels.
        ids: the subject ids, in the same order. Defaults to ``1..n``, which is PISC's own
            numbering (person ``k`` is ``bbox[k-1]``).
        thickness: outline width in pixels.

    Returns:
        A new ``(H, W, 3)`` uint8 RGB array with the overlay drawn on it.
    """
    import cv2

    canvas = np.ascontiguousarray(image_rgb.copy())
    height, width = canvas.shape[:2]
    labels = list(ids) if ids is not None else list(range(1, len(boxes) + 1))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    text_thickness = 2

    for index, (box, subject_id) in enumerate(zip(boxes, labels)):
        colour = BBOX_COLORS[index % len(BBOX_COLORS)]
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        text = f"P{subject_id}"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, text_thickness)
        # Above the box normally; inside it when the box starts at the top of the frame, so the
        # label never falls off-image and leaves a subject unnamed.
        label_top = y1 - text_h - baseline - 2
        if label_top < 0:
            label_top = min(y1 + 2, max(height - text_h - baseline - 2, 0))
        label_left = min(max(x1, 0), max(width - text_w - 6, 0))

        cv2.rectangle(
            canvas,
            (label_left, label_top),
            (label_left + text_w + 6, label_top + text_h + baseline + 2),
            colour,
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (label_left + 3, label_top + text_h + 1),
            font,
            scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return canvas
