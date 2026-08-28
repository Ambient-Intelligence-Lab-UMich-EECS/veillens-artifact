#!/usr/bin/env python3
"""
Download the model weights the baselines need, into $VEILLENS_MODELS.

    python scripts/fetch_baseline_weights.py --method all
    python scripts/fetch_baseline_weights.py --check          # verify, download nothing

Two of the five baselines need a checkpoint on disk:

    ocr-blur   frozen_east_text_detection.pb        92 MB
    mm-pgd     resnet18_places365.pth.tar           43 MB
               resnet50_places365.pth.tar           93 MB

The other three need no download here. `geoshield` pulls its three CLIP surrogates from
HuggingFace on first use (about 6 GB into $HF_HOME) and needs the `third_party/GeoShield`
submodule initialised; `gpt-image` and `rand-noise` need nothing at all.

Every file is checked against a SHA-256 recorded from the copies the reported results were
produced with, so a truncated download or a substituted mirror fails here rather than three hours
into a run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from veillens.paths import models_root  # noqa: E402


class Weight(NamedTuple):
    method: str
    relpath: str
    url: str
    sha256: str
    size: int


WEIGHTS: List[Weight] = [
    Weight(
        method="ocr-blur",
        relpath="east/frozen_east_text_detection.pb",
        # The EAST release is distributed as a frozen TensorFlow graph; this is the mirror
        # OpenCV's own text-detection sample points at.
        url="https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/"
            "frozen_east_text_detection.pb",
        sha256="9b486f3c3eee77b4c8cc91a83892c37026cca7d29b79bf3b93772ccd2db58454",
        size=96662756,
    ),
    Weight(
        method="mm-pgd",
        relpath="places365/resnet18_places365.pth.tar",
        url="http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        sha256="2f4759217d470da2b803f8f66cd4488a066406b555a5fb95ee9a4663f9f05588",
        size=45506139,
    ),
    Weight(
        method="mm-pgd",
        relpath="places365/resnet50_places365.pth.tar",
        url="http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
        sha256="46529c86902bd0cfb0ea562a30b2850c28d2620d96282b3db9c318e1d774f6c5",
        size=97270159,
    ),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(weight: Weight, destination: Path) -> None:
    """Fetch to a temporary name, then rename — so an interrupted run leaves no half file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    print(f"  downloading {weight.url}")
    print(f"          to  {destination}  ({weight.size / 1e6:.0f} MB)")
    # Carriage-return progress only on a terminal: redirected to a file it renders as one
    # enormous line, which is how a 90 MB download turns a readable log into noise.
    show_progress = sys.stdout.isatty()
    with urllib.request.urlopen(weight.url) as response, open(partial, "wb") as out:
        transferred = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            transferred += len(block)
            if show_progress:
                print(f"\r          {transferred / 1e6:7.0f} / {weight.size / 1e6:.0f} MB",
                      end="", flush=True)
    if show_progress:
        print()

    actual = sha256_of(partial)
    if actual != weight.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {weight.relpath}\n"
            f"  expected {weight.sha256}\n"
            f"  got      {actual}\n"
            "The file at that URL is not the one these results were produced with. "
            "Download it by hand and verify before using it."
        )
    partial.rename(destination)


def main(argv: List[str] | None = None) -> int:
    methods = sorted({weight.method for weight in WEIGHTS})
    parser = argparse.ArgumentParser(
        prog="fetch_baseline_weights.py",
        description=__doc__.split("\n\n")[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", default="all", choices=["all", *methods],
                        help="which baseline's weights to fetch (default: all)")
    parser.add_argument("--check", action="store_true",
                        help="report what is present and whether it verifies; download nothing")
    args = parser.parse_args(argv)

    root = models_root()
    wanted = [w for w in WEIGHTS if args.method in ("all", w.method)]

    print(f"\nmodel root: {root}")
    if not os.environ.get("VEILLENS_MODELS") and not os.environ.get("VEILLENS_DATA"):
        print("  (neither VEILLENS_MODELS nor VEILLENS_DATA is set, so this is the in-repo "
              "default;\n   see docs/DATA.md)")
    print()

    problems: Dict[str, str] = {}
    for weight in wanted:
        destination = root / weight.relpath
        label = f"{weight.method:10s} {weight.relpath}"

        if destination.is_file():
            actual = sha256_of(destination)
            if actual == weight.sha256:
                print(f"  ok       {label}")
                continue
            problems[weight.relpath] = "present but the checksum does not match"
            print(f"  BAD      {label}  (checksum mismatch)")
            if args.check:
                continue
            print("           re-downloading")
        elif args.check:
            print(f"  missing  {label}")
            problems[weight.relpath] = "missing"
            continue
        else:
            print(f"  fetch    {label}")

        try:
            download(weight, destination)
        except Exception as exc:  # noqa: BLE001 - urllib raises a wide family
            print(f"  FAILED   {label}: {exc}", file=sys.stderr)
            problems[weight.relpath] = str(exc)
            continue
        problems.pop(weight.relpath, None)
        print(f"  ok       {label}")

    if problems:
        print(f"\n{len(problems)} file(s) still unusable:", file=sys.stderr)
        for relpath, reason in problems.items():
            print(f"  - {relpath}: {reason}", file=sys.stderr)
        return 1

    print("\nall weights present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
