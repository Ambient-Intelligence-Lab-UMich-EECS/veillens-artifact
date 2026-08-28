"""
What a baseline is, in this repository.

A baseline is one function: an image goes in, a protected image comes out. Everything else —
loading the manifest, running the attacker over the result, writing a score table the same
`veillens-score` reads — is the runner's job, and is identical for all five. That split is
deliberate: the only thing that differs between the rows of the baseline table should be the
transform itself, so a difference in the numbers can only come from the method.

Five are shipped. Two are pixel transforms with no learned component (`rand-noise`, `ocr-blur`),
two are adversarial perturbations against a surrogate (`mm-pgd`, `geoshield`), and one hands the
whole job to an image-editing model (`gpt-image`).

Subclasses implement :meth:`apply`. They may also override :meth:`prepare` (one-off model
loading, or anything that needs to see the whole manifest first) and
:meth:`check_prerequisites` (weights and keys, checked before a run starts rather than an hour
in).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from PIL import Image

__all__ = ["Baseline"]


class Baseline(ABC):
    """One defense method, applied per image."""

    #: Registry key, and the directory name under ``baselines/`` that output lands in.
    name: str = ""

    #: One line for the run banner and the table in docs/BASELINES.md.
    summary: str = ""

    #: How many images to have in flight at once. Only the API-backed baseline raises this;
    #: the GPU ones stay at 1 because they are already saturating the device.
    workers: int = 1

    def __init__(self, **params: Any) -> None:
        self.params = params

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def check_prerequisites(cls, params: Dict[str, Any]) -> List[str]:
        """
        Human-readable reasons this baseline could not run, or an empty list.

        Checked from ``--dry-run`` and again before a real run starts. Must not import torch or
        touch a GPU: the point is to be answerable on a laptop.
        """
        return []

    def prepare(self, items: List[Dict[str, Any]]) -> None:
        """
        Called once before the first :meth:`apply`, with the resolved manifest.

        Load weights here rather than in ``__init__`` so that constructing a baseline stays free
        — the CLI builds one just to read its ``summary``.
        """

    def finish(self) -> None:
        """
        Called once after the last :meth:`apply`, whether or not the run succeeded.

        For releasing anything that registered itself somewhere global — forward hooks, in
        practice. Nothing here needs it for correctness; it exists so a run does not end with a
        library's ``__del__`` raising into interpreter shutdown, which reads like a failure.
        """

    @abstractmethod
    def apply(self, image_path: str, image_id: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Protect one image.

        Returns:
            The protected image at the source resolution, and a dict of per-image statistics
            that goes into ``metadata/<image_id>.json``. Anything in the dict must be
            JSON-serialisable.
        """
