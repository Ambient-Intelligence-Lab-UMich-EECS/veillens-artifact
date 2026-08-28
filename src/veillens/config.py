"""
Config loading and validation.

A pipeline config is a YAML file that may reference the roots from :mod:`veillens.paths` using
``${paths.repo_root}``, ``${paths.data}``, ``${paths.runs}``, ``${paths.models}`` and
``${paths.gsam2}``. :func:`load_config` substitutes them, so a config never contains a
machine-specific path.

    from veillens.config import load_config
    cfg = load_config("configs/geolocation/base.yaml")

Substitution is plain string replacement rather than OmegaConf interpolation. OmegaConf is
available (SAM 2 pulls it in) but making it a hard requirement of config loading would mean the
config layer could not be imported without the full GPU stack installed, which breaks
``--dry-run`` on a laptop. The placeholder set is tiny and closed, so a dict-driven replace is
enough and keeps this module dependency-light.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from veillens import paths
from veillens.privacy.registry import canonical_name
from veillens.verify.granularity import CLUE_LEVEL_TO_RUNG, target_rung

__all__ = ["load_config", "validate_config", "substitute_paths", "REQUIRED_SECTIONS"]

#: Sections every pipeline config must define.
REQUIRED_SECTIONS = ("dataset", "experiment", "models", "output", "evaluation")

_PLACEHOLDER = re.compile(r"\$\{paths\.(\w+)\}")


def _roots() -> Dict[str, str]:
    return {
        "repo_root": str(paths.REPO_ROOT),
        "data": str(paths.data_root()),
        "runs": str(paths.runs_root()),
        "models": str(paths.models_root()),
        # Normally <repo>/third_party/Grounded-SAM-2, but it is a submodule people commonly
        # already have checked out elsewhere, so VEILLENS_GSAM2 redirects it. Exposed as a
        # placeholder so a config can name it without hardcoding either location.
        "gsam2": str(paths.grounded_sam2_root()),
        # Same story for GeoShield, which only the baselines need.
        "geoshield": str(paths.geoshield_root()),
    }


def substitute_paths(node: Any, roots: Mapping[str, str] | None = None) -> Any:
    """Recursively replace ``${paths.X}`` placeholders in every string in *node*."""
    table = dict(roots) if roots is not None else _roots()

    def _sub_one(text: str) -> str:
        def repl(match: re.Match) -> str:
            key = match.group(1)
            if key not in table:
                raise KeyError(
                    f"Unknown path placeholder ${{paths.{key}}}. "
                    f"Known roots: {sorted(table)}"
                )
            return table[key]

        return _PLACEHOLDER.sub(repl, text)

    if isinstance(node, str):
        return _sub_one(node)
    if isinstance(node, list):
        return [substitute_paths(item, table) for item in node]
    if isinstance(node, dict):
        return {key: substitute_paths(value, table) for key, value in node.items()}
    return node


def validate_config(config: Dict[str, Any], *, check_dataset: bool = True) -> None:
    """
    Check that a config has the sections the pipeline needs.

    Args:
        config: an already-loaded config dict.
        check_dataset: also require the dataset manifest to exist on disk. Turn this off for
            ``--dry-run`` style checks on a machine that has the code but not the data.

    Raises:
        ValueError: a required section is missing, or ``experiment`` carries a key that no
            longer means anything.
        FileNotFoundError: the dataset manifest is absent and *check_dataset* is set.
    """
    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(
            f"Config is missing required section(s): {', '.join(missing)}. "
            f"Expected all of: {', '.join(REQUIRED_SECTIONS)}"
        )

    _validate_experiment(config.get("experiment") or {})

    if check_dataset:
        manifest = Path(config["dataset"]["dataset_json_file"])
        if not manifest.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found: {manifest}\n"
                "Set VEILLENS_DATA to where your datasets live, or see docs/DATA.md."
            )


def _validate_experiment(experiment: Mapping[str, Any]) -> None:
    """
    Reject experiment keys that would silently do nothing.

    A config key that is documented but dead is a reproducibility failure, not a harmless
    leftover: it tells a reader the run was controlled by something that never ran. So a
    superseded key is an error at load, with the migration in the message, rather than a value
    quietly ignored at runtime.
    """
    if "val_dist" in experiment:
        raise ValueError(
            "experiment.val_dist is no longer read. The loop now stops at the threshold derived "
            "from experiment.mask_level: the coarsest protected cue granularity picks a rung on "
            "the ladder, and the rung fixes the distance (city-level protection => 10 km).\n"
            "Remove the key. See docs/VERIFICATION.md."
        )

    mask_level = experiment.get("mask_level")
    if mask_level is None:
        raise ValueError(
            "experiment.mask_level is required: it is what the run removes and what "
            "verification checks. See docs/VERIFICATION.md."
        )

    privacy_class = canonical_name(experiment.get("privacy_class", "geolocation"))
    if privacy_class == "geolocation" and target_rung(privacy_class, mask_level) is None:
        raise ValueError(
            f"experiment.mask_level={mask_level!r} names no known cue granularity, so no "
            f"verification rung can be derived from it. Expected a list drawn from: "
            f"{', '.join(CLUE_LEVEL_TO_RUNG)}."
        )


def load_config(config_path: str | Path, *, validate: bool = True,
                check_dataset: bool = True) -> Dict[str, Any]:
    """
    Load a pipeline config, resolving ``${paths.*}`` placeholders.

    Relative *config_path* values are taken as repo-relative, so
    ``load_config("configs/geolocation/base.yaml")`` works from any directory.
    """
    resolved = paths.resolve(config_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")

    with open(resolved, "r") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(config).__name__}: {resolved}")

    config = substitute_paths(config)
    config["_config_path"] = str(resolved)

    if validate:
        validate_config(config, check_dataset=check_dataset)
    return config
