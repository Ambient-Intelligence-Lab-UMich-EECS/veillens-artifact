"""
Dataset manifest format, and the loader that makes it portable.

**Schema v2** (current) stores a dataset name and a relative path, joined against
``$VEILLENS_DATA`` when the manifest is read::

    {
      "schema_version": 2,
      "items": [
        {"image_id": "2017_10003252", "dataset": "vispr/test2017",
         "relpath": "2017_10003252.jpg", "ground_truth_score": 0.795}
      ]
    }

Storing ``dataset`` + ``relpath`` rather than an absolute path is what makes a manifest portable
between machines. :func:`load_manifest` hands back items carrying a resolved absolute
``image_path``, so callers never join paths themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from veillens import paths

__all__ = ["load_manifest", "resolve_item_path", "SCHEMA_VERSION"]

SCHEMA_VERSION = 2


def resolve_item_path(item: Dict[str, Any], data_root: Path | None = None) -> Path:
    """
    Absolute image path for a manifest item, whichever schema it came from.

    Raises:
        KeyError: the item has no ``dataset``/``relpath`` pair.
    """
    root = data_root if data_root is not None else paths.data_root()
    try:
        return root / item["dataset"] / item["relpath"]
    except KeyError:
        raise KeyError(
            f"Manifest item {item.get('image_id', '<no id>')!r} has no usable path: "
            "every item needs a 'dataset' and a 'relpath'."
        ) from None


def load_manifest(manifest_path: str | Path,
                  data_root: Path | None = None) -> List[Dict[str, Any]]:
    """
    Load a dataset manifest and resolve every item's ``image_path``.

    Args:
        manifest_path: path to the manifest; relative paths are repo-relative.
        data_root: override for ``$VEILLENS_DATA``, mostly for tests.

    Returns:
        The list of items, each with an absolute ``image_path`` string added. Item keys are
        otherwise untouched — in particular ``image_id`` is preserved, because cached VLM
        responses are keyed on it.
    """
    path = paths.resolve(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {path}\n"
            "Point VEILLENS_DATA at your data directory, or see docs/DATA.md."
        )

    raw = json.loads(path.read_text())
    if not (isinstance(raw, dict) and "items" in raw):
        raise ValueError(
            f"{path} is not a v{SCHEMA_VERSION} manifest: expected a mapping with an 'items' key."
        )
    items = raw["items"]

    root = data_root if data_root is not None else paths.data_root()
    out: List[Dict[str, Any]] = []
    for item in items:
        item = dict(item)
        item["image_path"] = str(resolve_item_path(item, root))
        out.append(item)
    return out
