"""
Path resolution for VeilLens.

Two kinds of location, resolved two different ways:

* **Anything under version control** — code, configs, prompts — is found relative to the
  repository root. It is never configurable, because cloning the repo must be enough to find it.
* **Anything heavy** — datasets, model weights, run outputs — is configurable, because it has to
  live on scratch or a scratch-like volume while the clone lives somewhere else.

Precedence for the heavy roots: an explicit argument, then the environment variable, then a
sensible default inside the repo.

    VEILLENS_DATA    datasets and model weights   (default: <repo>/data)
    VEILLENS_RUNS    pipeline output              (default: <repo>/runs)
    VEILLENS_MODELS  checkpoints                  (default: $VEILLENS_DATA/models)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "RUN_SUBDIRS",
    "INTERMEDIATE_DIR",
    "SEGMENTATION_DIR",
    "FINAL_DIR",
    "REPO_ROOT",
    "PROMPTS",
    "CONFIGS",
    "EXAMPLES",
    "data_root",
    "runs_root",
    "models_root",
    "resolve",
    "bootstrap_grounded_sam2",
]


#: Run-level subdirectories under ``output.base_dir``.
RUN_SUBDIRS = {
    "original_cache": "original_cache",
    "eval_cache": "eval_cache",
    "evaluation_results": "evaluation_results",
    "phase_cache": "phase_cache",
}

#: Per-image output lives under ``<base_dir>/<INTERMEDIATE_DIR>/<image_name>/``.
INTERMEDIATE_DIR = "intermediate"

#: Per-image subdirectories under ``intermediate/<image_name>/``.
SEGMENTATION_DIR = "segmentation"
FINAL_DIR = "final"


def _find_repo_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file until *marker* is found; fall back to two levels up."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / marker).is_file():
            return candidate
    # src/veillens/paths.py -> src/veillens -> src -> <repo>
    return here.parents[2]


REPO_ROOT: Path = _find_repo_root()
PROMPTS: Path = REPO_ROOT / "prompts"
CONFIGS: Path = REPO_ROOT / "configs"
EXAMPLES: Path = REPO_ROOT / "examples"


def data_root() -> Path:
    return Path(os.environ.get("VEILLENS_DATA", REPO_ROOT / "data"))


def runs_root() -> Path:
    return Path(os.environ.get("VEILLENS_RUNS", REPO_ROOT / "runs"))


def models_root() -> Path:
    return Path(os.environ.get("VEILLENS_MODELS", data_root() / "models"))


def resolve(value: str | os.PathLike) -> Path:
    """
    Turn a config path into an absolute one.

    Absolute paths pass through. Relative paths are taken as repo-relative, which is what every
    config in this repo means by a bare path like ``prompts/geolocation/schema_original.json``.
    """
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def grounded_sam2_root() -> Path:
    """
    Where the Grounded-SAM-2 submodule lives.

    Overridable with ``VEILLENS_GSAM2`` for the case where someone has it installed elsewhere.
    """
    override = os.environ.get("VEILLENS_GSAM2")
    if override:
        return Path(override)
    return REPO_ROOT / "third_party" / "Grounded-SAM-2"


def bootstrap_grounded_sam2() -> Path:
    """
    Put Grounded-SAM-2 on ``sys.path`` so ``sam2`` and ``grounding_dino`` import.

    Upstream ships no installable package, so the path has to be prepended before those imports
    run. Idempotent; safe to call from several modules.

    Raises:
        FileNotFoundError: if the submodule is missing, with the command to fix it.
    """
    root = grounded_sam2_root()
    if not root.exists():
        raise FileNotFoundError(
            f"Grounded-SAM-2 not found at {root}.\n"
            "Initialise the submodule:\n"
            "    git submodule update --init --recursive third_party/Grounded-SAM-2\n"
            "or point VEILLENS_GSAM2 at an existing checkout."
        )
    for entry in (root, root / "grounding_dino"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root
