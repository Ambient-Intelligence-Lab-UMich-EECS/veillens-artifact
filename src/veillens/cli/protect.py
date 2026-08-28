#!/usr/bin/env python3
"""
Run the VeilLens protection loop over a dataset.

    veillens-protect --config configs/geolocation.yaml
    veillens-protect --config configs/geolocation.yaml --dry-run

``--dry-run`` resolves and validates everything the run would need — config schema, prompt and
schema files, model checkpoints, the dataset manifest — and then exits without making an API
call or touching a GPU. It is the fastest way to find out whether an install is complete.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from veillens.config import load_config, validate_config
from veillens.paths import resolve
from veillens.privacy.spec import get_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veillens-protect",
        description="Reason about, ground, edit and verify privacy cues in a set of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # check an install without spending anything
  veillens-protect --config configs/geolocation.yaml --dry-run

  # protect the compositional-privacy set
  veillens-protect --config configs/compositional.yaml

the loop, per image:
  reason    a VLM oracle lists the cues that leak the target attribute, and why
  ground    each cue is localised to a mask (Grounding DINO + SAM 2)
  edit      selected cues are inpainted away (SDXL + ControlNet)
  verify    the oracle re-reads the result; if it still leaks, iterate
            (bounded by experiment.max_iter and experiment.val_dist)
""",
    )
    parser.add_argument("--config", required=True,
                        help="pipeline config YAML (see configs/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate config, prompts, schemas and paths, then exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _check_referenced_paths(config: dict) -> list[str]:
    """Return human-readable problems with everything the config points at."""
    problems: list[str] = []

    spec = get_spec(config)
    for label, rel in (
        ("round1 prompt", spec.round1_prompt_path),
        ("round2 prompt", spec.round2_prompt_path),
        ("eval prompt", spec.eval_prompt_path),
        ("round1 schema", spec.round1_schema_path),
        ("round2 schema", spec.round2_schema_path),
        ("eval schema", spec.eval_schema_path),
    ):
        if not resolve(rel).is_file():
            problems.append(f"missing {label}: {rel}")

    models = config.get("models", {})
    for key in ("sam2_checkpoint", "grounding_dino_checkpoint"):
        value = models.get(key)
        if value and not Path(value).exists():
            problems.append(f"missing {key}: {value}  (see docs/DATA.md)")
    root = models.get("grounded_sam2_root")
    if root and not Path(root).is_dir():
        problems.append(
            f"missing grounded_sam2_root: {root}\n"
            "      git submodule update --init --recursive third_party/Grounded-SAM-2"
        )
    elif root:
        # GroundingDINO's deformable-attention kernel is a compiled extension, not Python. A
        # checkout without it imports fine and loads weights fine, then raises
        # `name '_C' is not defined` on the first detection -- which the loop catches per cue and
        # logs as "no ground". A whole run then completes, reports success, and writes a score
        # table computed from images nothing ever edited. Check for the built object here, where
        # it costs nothing, rather than discovering it an hour and several hundred API calls in.
        gdino = Path(root) / "grounding_dino"
        if not list((gdino / "groundingdino").glob("_C*.so")):
            problems.append(
                "GroundingDINO CUDA extension is not built (no _C*.so in\n"
                f"      {gdino / 'groundingdino'}).\n"
                "      Detection silently returns nothing without it. Build it with:\n"
                f"      pip install --no-build-isolation -e {gdino}\n"
                "      (needs nvcc and CUDA_HOME; see third_party/Grounded-SAM-2/INSTALL.md)"
            )
    manifest = Path(config["dataset"]["dataset_json_file"])
    if not manifest.exists():
        problems.append(f"missing dataset manifest: {manifest}  (see docs/DATA.md)")

    return problems


def _summarise(config: dict) -> None:
    experiment = config.get("experiment", {})
    print("\n" + "=" * 66)
    print("VeilLens — run configuration")
    print("=" * 66)
    print(f"  privacy class : {experiment.get('privacy_class', 'geolocation')}")
    print(f"  mask level    : {experiment.get('mask_level')}")
    # max_iter counts R2 iterations; R1's own inpainting iteration adds one more.
    print(f"  max rounds    : {int(experiment.get('max_iter', 1)) + 1}")
    val_dist = experiment.get('val_dist')
    print(f"  stop when     : error >= {val_dist} km" if val_dist is not None
          else "  stop when     : max rounds reached (no distance-based stop)")
    print(f"  dataset       : {config['dataset']['dataset_json_file']}")
    evaluation = config.get('evaluation', {})
    reasoning = config.get('reasoning') or evaluation
    print(f"  reasoning     : {reasoning.get('model')}   (defense oracle)")
    print(f"  evaluation    : {evaluation.get('model')}   (attacker)")
    print(f"  output        : {config['output']['base_dir']}")
    print("=" * 66 + "\n")


async def _run(config: dict) -> None:
    # Imported here, not at module scope: the heavy GPU stack should not be required for
    # --help or --dry-run.
    from veillens.loop.driver import GeoLocationPipeline

    pipeline = GeoLocationPipeline(config)
    await pipeline.run()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = load_config(args.config, validate=False)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        validate_config(config, check_dataset=False)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _summarise(config)

    problems = _check_referenced_paths(config)
    if args.dry_run:
        if problems:
            print("dry run found problems:\n")
            for problem in problems:
                print(f"  - {problem}")
            print("\nConfig itself is valid; the items above must exist before a real run.")
            return 1
        print("dry run OK — config valid and every referenced file is present.")
        return 0

    if problems:
        print("cannot start; missing prerequisites:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\nRe-run with --dry-run after fixing, or see docs/DATA.md.", file=sys.stderr)
        return 1

    started = time.time()
    try:
        asyncio.run(_run(config))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        # The loop raises this when a run finished but produced nothing usable -- a result that
        # must not be reported as success just because every stage returned.
        print(f"\nrun failed: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - started
    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\ndone in {hours}h {minutes}m {seconds}s")
    print(f"results: {config['output']['base_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
