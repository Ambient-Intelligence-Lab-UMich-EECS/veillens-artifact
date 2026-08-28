"""
`rand-noise` — Gaussian blur applied to a random scatter of pixels.

The weakest defense in the table and the cheapest: no model, no weights, no GPU. It blurs the
whole frame once, then copies a uniformly random 10% of the blurred pixels back over the
original. The damage is spread as isolated speckles rather than regions, which is why it costs
far more perceptually than its 10% pixel budget suggests (SSIM 0.957, LPIPS 0.143) while leaving
most geolocation evidence readable.

Included because it is the control: any method that does not beat scattered blur at the same
pixel budget is not exploiting structure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from veillens.baselines.base import Baseline

__all__ = ["RandomPixelBlur"]


class RandomPixelBlur(Baseline):
    name = "rand-noise"
    summary = "Gaussian blur on a random 10% of pixels"

    def __init__(self, pixel_ratio: float = 0.1, blur_ksize: int = 51,
                 blur_sigma: float = 0.0, seed: int = 42, **params: Any) -> None:
        super().__init__(pixel_ratio=pixel_ratio, blur_ksize=blur_ksize,
                         blur_sigma=blur_sigma, seed=seed, **params)
        self.pixel_ratio = pixel_ratio
        # cv2.GaussianBlur rejects an even kernel; round up rather than raise, since the
        # difference between 50 and 51 is not something a config author means to express.
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.blur_sigma = blur_sigma
        self.seed = seed

    def apply(self, image_path: str, image_id: str) -> Tuple[Image.Image, Dict[str, Any]]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")

        height, width = image.shape[:2]
        total_pixels = height * width

        # Per-image seed derived from the run seed and the image id, so the scatter is
        # reproducible and yet different for every frame. zlib.crc32 rather than hash():
        # Python's string hash is salted per process unless PYTHONHASHSEED is set, so hash()
        # would make the "reproducible" pattern change on every invocation.
        from zlib import crc32
        rng = np.random.default_rng((self.seed, crc32(image_id.encode())))

        blurred = cv2.GaussianBlur(image, (self.blur_ksize, self.blur_ksize), self.blur_sigma)

        n_blur = int(total_pixels * self.pixel_ratio)
        flat = rng.choice(total_pixels, size=n_blur, replace=False)
        rows, cols = flat // width, flat % width

        result = image.copy()
        result[rows, cols] = blurred[rows, cols]

        stats = {
            "modified_pixels": int(n_blur),
            "total_pixels": int(total_pixels),
            "modified_ratio": float(n_blur / total_pixels),
            "blur_ksize": self.blur_ksize,
            "blur_sigma": self.blur_sigma or "auto",
        }
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), stats
