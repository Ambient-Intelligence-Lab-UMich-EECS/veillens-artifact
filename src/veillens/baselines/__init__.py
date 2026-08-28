"""
The five prior-work defenses the paper compares against.

    from veillens.baselines import build, available
    method = build("rand-noise", {"pixel_ratio": 0.1})

Each is a :class:`~veillens.baselines.base.Baseline`: one image in, one protected image out.
:mod:`veillens.baselines.runner` does everything around them — manifest, output layout, and the
attacker pass that turns protected images into the same `evaluation_results.json` the protection
loop writes, so `veillens-score` reads a baseline run and a VeilLens run identically.

Entries are imported lazily. Two of the five need torch and one needs an OpenAI key; importing
this package must stay free of both, because `--dry-run` and the test suite have to work on a
laptop.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Type

from veillens.baselines.base import Baseline

__all__ = ["REGISTRY", "available", "get_baseline_class", "build", "Baseline"]

#: Method name -> (module, class name). The name is also the output directory name.
REGISTRY: Dict[str, tuple] = {
    "rand-noise": ("veillens.baselines.rand_noise", "RandomPixelBlur"),
    "ocr-blur": ("veillens.baselines.ocr_blur", "OCRTextBlur"),
    "mm-pgd": ("veillens.baselines.mm_pgd", "MultiModelPGD"),
    "geoshield": ("veillens.baselines.geoshield", "GeoShield"),
    "gpt-image": ("veillens.baselines.gpt_image", "GPTImageDefense"),
}


def available() -> List[str]:
    """Registered method names, in the order the docs table lists them."""
    return list(REGISTRY)


def get_baseline_class(name: str) -> Type[Baseline]:
    """
    The class implementing *name*.

    Raises:
        ValueError: no such method.
    """
    try:
        module_name, class_name = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown baseline {name!r}. Available: {', '.join(available())}"
        ) from None
    return getattr(importlib.import_module(module_name), class_name)


def build(name: str, params: Dict[str, Any] | None = None) -> Baseline:
    """Construct method *name* from its `configs/baselines.yaml` section."""
    return get_baseline_class(name)(**(params or {}))
