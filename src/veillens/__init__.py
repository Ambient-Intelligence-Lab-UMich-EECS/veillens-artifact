"""
VeilLens — explainable mitigation of VLM-based visual privacy risk.

The package mirrors the four stages of the protection loop:

    reason/   a VLM oracle says what is inferable from the image, and why
    ground/   those language cues become pixel masks (open-set detection + SAM 2)
    edit/     selected cues are removed by ControlNet-guided SDXL inpainting

Stage 4, verification, has no directory of its own: the oracle that re-reads the edited image
is the same `reason/` client, driven by the iteration gate in `loop/api_workers.py`.
`loop/` orchestrates the four stages; `privacy/` holds the per-setting specs that make the
loop generic across privacy classes.
"""

__version__ = "0.1.0"
