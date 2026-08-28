"""
Stage 3 (editing), model container.

`SDXLInpaintingPipeline` loads the three GPU components once -- Grounded-SAM-2, the depth
extractor, and the ControlNet inpainter -- and holds them together so the loop can hand the same
loaded models to both the segmentation thread and the batched diffusion path. Construction is the
whole interface: callers reach through to `.segmenter`, `.extractor` and `.inpainter` directly.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from veillens.edit.control import DepthExtractor
from veillens.edit.inpaint import ControlNetInpainter
from veillens.ground.segmenter import GroundedSAM2Segmenter


class SDXLInpaintingPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: an already-loaded config dict, as returned by
                ``veillens.config.load_config``. It must be the loaded dict rather than a path:
                loading it here would skip ``substitute_paths``, and every ``${paths.*}`` in the
                model section would reach torch as a literal string.
        """
        self.config = config

        # Initialize components
        print("Initializing Grounded-SAM-2 Segmenter...")
        self.segmenter = GroundedSAM2Segmenter(self.config)

        print("Initializing Depth Extractor...")
        self.extractor = DepthExtractor(self.config)

        print("Initializing ControlNet Inpainter...")
        self.inpainter = ControlNetInpainter(self.config)

        # Clear GPU cache to free fragmented memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ GPU cache cleared")

        print("Pipeline initialization complete!\n")


