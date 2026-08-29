"""
`ocr-blur` — detect text with EAST, blur inside the boxes.

Needs one weight file, `frozen_east_text_detection.pb` (92 MB, `scripts/fetch_baseline_weights.py`).
It runs on CPU through OpenCV's DNN module; no GPU, no torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from veillens.baselines.base import Baseline

__all__ = ["OCRTextBlur", "decode_east"]

#: EAST's feature maps are 1/4 the input resolution.
_EAST_STRIDE = 4.0

#: Mean subtraction from the original EAST release. Not a knob.
_EAST_MEAN = (123.68, 116.78, 103.94)

#: The two output layers, by name, in the frozen graph.
_EAST_LAYERS = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]


def decode_east(scores: np.ndarray, geometry: np.ndarray,
                score_thresh: float = 0.5) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """
    Turn EAST's score and geometry maps into axis-aligned boxes.

    Returns:
        Boxes as ``(start_x, start_y, end_x, end_y)`` in EAST input coordinates, and their
        confidences. Rotation is discarded: the boxes only ever become a blur mask, so the
        axis-aligned hull is enough and keeps the mask arithmetic trivial.
    """
    num_rows, num_cols = scores.shape[2:4]
    rects: List[Tuple[int, int, int, int]] = []
    confidences: List[float] = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0, x1, x2, x3 = (geometry[0, i, y] for i in range(4))
        angles = geometry[0, 4, y]

        for x in range(num_cols):
            score = scores_data[x]
            if score < score_thresh:
                continue

            cos_a, sin_a = np.cos(angles[x]), np.sin(angles[x])
            height = x0[x] + x2[x]
            width = x1[x] + x3[x]

            offset_x, offset_y = x * _EAST_STRIDE, y * _EAST_STRIDE
            end_x = int(offset_x + cos_a * x1[x] + sin_a * x2[x])
            end_y = int(offset_y - sin_a * x1[x] + cos_a * x2[x])

            rects.append((int(end_x - width), int(end_y - height), end_x, end_y))
            confidences.append(float(score))

    return rects, confidences


class OCRTextBlur(Baseline):
    name = "ocr-blur"
    summary = "EAST text detection, then Gaussian blur inside the detected boxes"

    def __init__(self, east_model: str = "", max_ratio: float = 0.4,
                 conf_threshold: float = 0.2, nms_threshold: float = 0.3,
                 input_width: int = 1920, input_height: int = 1920,
                 blur_ksize: int = 31, box_padding: int = 8, **params: Any) -> None:
        super().__init__(east_model=east_model, max_ratio=max_ratio,
                         conf_threshold=conf_threshold, nms_threshold=nms_threshold,
                         input_width=input_width, input_height=input_height,
                         blur_ksize=blur_ksize, box_padding=box_padding, **params)
        self.east_model = east_model
        self.max_ratio = max_ratio
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.box_padding = box_padding
        self.net = None

    @classmethod
    def check_prerequisites(cls, params: Dict[str, Any]) -> List[str]:
        model = params.get("east_model")
        if not model or not Path(model).is_file():
            return [
                f"missing EAST text detector: {model or '<unset>'}\n"
                "      python scripts/fetch_baseline_weights.py --method ocr-blur"
            ]
        return []

    def prepare(self, items: List[Dict[str, Any]]) -> None:
        self.net = cv2.dnn.readNet(self.east_model)

    def apply(self, image_path: str, image_id: str) -> Tuple[Image.Image, Dict[str, Any]]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")

        height, width = image.shape[:2]
        total_pixels = height * width

        def unchanged(num_boxes: int = 0) -> Tuple[Image.Image, Dict[str, Any]]:
            return (Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                    {"num_boxes": num_boxes, "modified_pixels": 0, "modified_ratio": 0.0,
                     "total_pixels": int(total_pixels)})

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (self.input_width, self.input_height)),
            scalefactor=1.0, size=(self.input_width, self.input_height),
            mean=_EAST_MEAN, swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        scores, geometry = self.net.forward(_EAST_LAYERS)

        rects, confidences = decode_east(scores, geometry, score_thresh=self.conf_threshold)
        keep = cv2.dnn.NMSBoxes(bboxes=rects, scores=confidences,
                                score_threshold=self.conf_threshold,
                                nms_threshold=self.nms_threshold)
        if len(keep) == 0:
            return unchanged()

        # Map from EAST's square input back to the source frame, pad, and clip.
        scale_x, scale_y = width / float(self.input_width), height / float(self.input_height)
        boxes = []
        for i in np.asarray(keep).flatten():
            x1, y1, x2, y2 = rects[i]
            x1 = max(int(x1 * scale_x) - self.box_padding, 0)
            y1 = max(int(y1 * scale_y) - self.box_padding, 0)
            x2 = min(int(x2 * scale_x) + self.box_padding, width - 1)
            y2 = min(int(y2 * scale_y) + self.box_padding, height - 1)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

        if not boxes:
            return unchanged()

        # Smallest boxes first, so a pixel budget spends itself on many small signs rather than
        # one billboard. Boxes overlap, so the budget is measured on the accumulated mask
        # instead of by summing areas.
        max_pixels = int(total_pixels * self.max_ratio)
        mask = np.zeros((height, width), dtype=np.uint8)
        used = 0
        selected = 0
        for x1, y1, x2, y2 in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])):
            candidate = mask.copy()
            cv2.rectangle(candidate, (x1, y1), (x2, y2), 255, thickness=-1)
            # int(): np.count_nonzero returns np.int64, which json.dump refuses. Left as-is it
            # reaches the per-image metadata file and fails the write half way through.
            grown = int(np.count_nonzero(candidate))
            if grown <= max_pixels:
                mask, used, selected = candidate, grown, selected + 1

        if used == 0:
            return unchanged()

        blurred = cv2.GaussianBlur(image, (self.blur_ksize, self.blur_ksize), 0)
        result = image.copy()
        painted = mask > 0
        result[painted] = blurred[painted]

        stats = {
            "num_boxes": selected,
            "num_boxes_detected": len(boxes),
            "modified_pixels": int(used),
            "modified_ratio": float(used / total_pixels),
            "total_pixels": int(total_pixels),
        }
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), stats
