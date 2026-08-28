"""Stage 3 — editing: remove selected cues with controllable generative inpainting."""

from veillens.edit.control import DepthExtractor
from veillens.edit.inpaint import ControlNetInpainter
from veillens.edit.pipeline import SDXLInpaintingPipeline

__all__ = ["DepthExtractor", "ControlNetInpainter", "SDXLInpaintingPipeline"]
