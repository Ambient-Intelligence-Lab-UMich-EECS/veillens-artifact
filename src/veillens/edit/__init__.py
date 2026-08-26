"""Stage 3 — editing: remove selected cues with controllable generative inpainting."""

from veillens.edit.control import EdgeDepthExtractor
from veillens.edit.inpaint import ControlNetInpainter
from veillens.edit.pipeline import SDXLInpaintingPipeline

__all__ = ["EdgeDepthExtractor", "ControlNetInpainter", "SDXLInpaintingPipeline"]
