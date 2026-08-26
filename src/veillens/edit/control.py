"""
Stage 3 (editing), conditioning half.

`DepthExtractor` produces the ControlNet conditioning image — a MiDaS depth map — that keeps an
inpainted region structurally consistent with its surroundings.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np
import torch


class DepthExtractor:
    """Extracts MiDaS depth maps from images."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config['detection']['device']

        # Load MiDaS model and transforms from torch.hub (optimized single load)
        print(f"\nLoading MiDaS model and transforms: {config['models']['midas_model']}...")
        print("      (This may take 1-3 minutes on networked storage - please wait...)")
        midas_start = time.time()

        # Load hub repo once with force_reload=False to use cache
        midas_repo = torch.hub.load("intel-isl/MiDaS", config['models']['midas_model'], force_reload=False)
        self.midas = midas_repo
        self.midas.to(self.device)
        self.midas.eval()
        print(f"      ✓ MiDaS model loaded in {time.time() - midas_start:.1f}s")

        # Load transforms from the same cached repo (reuses already loaded hub)
        print("      Loading MiDaS transforms...")
        transform_start = time.time()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", force_reload=False)
        print(f"      ✓ MiDaS transforms loaded in {time.time() - transform_start:.1f}s")
        if config['models']['midas_model'] in ["DPT_Large", "DPT_Hybrid"]:
            self.midas_transform = midas_transforms.dpt_transform
        else:
            self.midas_transform = midas_transforms.small_transform

    def extract_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Extract depth map using MiDaS.

        Args:
            image: Input image in RGB format (H, W, 3)

        Returns:
            Depth map normalized to 0-255 (H, W)
        """
        # Prepare input
        input_batch = self.midas_transform(image).to(self.device)

        # Predict depth
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()
        return cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
