"""
VeilLens — explainable mitigation of VLM-based visual privacy risk.

The package mirrors the four stages of the protection loop:

    reason/   a VLM oracle says what is inferable from the image, and why
    ground/   those language cues become pixel masks (open-set detection + SAM 2)
    edit/     selected cues are removed by ControlNet-guided SDXL inpainting
    verify/   the oracle re-reads the edited image and decides whether to iterate

`loop/` orchestrates them; `privacy/` holds the per-setting specs that make the loop generic.
"""

__version__ = "0.1.0"
