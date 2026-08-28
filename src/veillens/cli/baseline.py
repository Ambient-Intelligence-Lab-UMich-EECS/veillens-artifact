#!/usr/bin/env python3
"""
Run a prior-work baseline over the same dataset the protection loop uses.

    veillens-baseline --method rand-noise --config configs/geolocation.yaml
    veillens-baseline --method all --config configs/geolocation.yaml --limit 4
    veillens-baseline --method mm-pgd --config configs/geolocation.yaml --dry-run

The pipeline config supplies the dataset, the attacker and the output root; the method's own
hyperparameters come from `configs/baselines.yaml`. That split is what makes the comparison a
comparison: every row of the baseline table is measured on the same images by the same attacker,
and the only thing that varies is the defense.

Output lands in `<output.base_dir>/baselines/<method>/`, with an
`evaluation_results/evaluation_results.json` in the same format the loop writes — so
`veillens-score <that directory>` works with no flags.

`--dry-run` resolves the config, the manifest and every weight file the chosen methods need, then
exits without touching a GPU or making an API call.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from veillens.baselines import available, get_baseline_class
from veillens.config import load_config, validate_config
from veillens.privacy.spec import get_spec

DEFAULT_BASELINE_CONFIG = "configs/baselines.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veillens-baseline",
        description="Run a prior-work defense on the same images, scored by the same attacker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
methods:
  rand-noise   Gaussian blur on a random 10%% of pixels                 no weights, no GPU
  ocr-blur     EAST text detection, then blur inside the boxes          EAST .pb, no GPU
  mm-pgd       Union-CAM-masked PGD, Places365 ResNet-18 + ResNet-50    2 checkpoints, GPU
  geoshield    100-step ensemble PGD over three CLIP surrogates         submodule, GPU
  gpt-image    one gpt-image-2 edit per frame                           OPENAI_API_KEY, paid

examples:
  # check every method's prerequisites without spending anything
  veillens-baseline --method all --config configs/geolocation.yaml --dry-run

  # four images through the cheapest baseline, then score it
  veillens-baseline --method rand-noise --config configs/geolocation.yaml --limit 4
  veillens-score runs/geolocation/baselines/rand-noise
""",
    )
    parser.add_argument("--method", required=True,
                        help=f"one of: {', '.join(available())}, or 'all'")
    parser.add_argument("--config", required=True,
                        help="pipeline config YAML (the same file veillens-protect takes)")
    parser.add_argument("--baselines", default=DEFAULT_BASELINE_CONFIG,
                        help=f"baseline hyperparameters (default: {DEFAULT_BASELINE_CONFIG})")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="process only the first N manifest entries")
    parser.add_argument("--no-resume", action="store_true",
                        help="re-protect images that already have output")
    parser.add_argument("--no-eval", action="store_true",
                        help="write protected images and stop; skip the attacker pass")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate config, manifest and weights, then exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _resolve_methods(requested: str) -> List[str]:
    if requested == "all":
        return available()
    if requested not in available():
        raise ValueError(f"Unknown baseline {requested!r}. Available: {', '.join(available())}, all")
    return [requested]


def _fill_privacy_class(params: Dict[str, Any], privacy_class: str) -> Dict[str, Any]:
    """
    Expand `{privacy_class}` in string values.

    One method — `gpt-image` — needs a different prompt per privacy class, and that is the whole
    reason this exists. Substitution is opt-in per value rather than a blanket `.format()` so a
    parameter that legitimately contains braces is left alone.
    """
    return {
        key: value.format(privacy_class=privacy_class)
        if isinstance(value, str) and "{privacy_class}" in value else value
        for key, value in params.items()
    }


def _check(config: Dict[str, Any], methods: List[str],
           sections: Dict[str, Dict[str, Any]]) -> List[str]:
    """Everything that would stop these methods from running, as readable lines."""
    problems: List[str] = []

    manifest = Path(config["dataset"]["dataset_json_file"])
    if not manifest.exists():
        problems.append(f"missing dataset manifest: {manifest}  (see docs/DATA.md)")

    for method in methods:
        for problem in get_baseline_class(method).check_prerequisites(sections[method]):
            problems.append(f"[{method}] {problem}")
    return problems


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        methods = _resolve_methods(args.method)
        config = load_config(args.config, validate=False)
        validate_config(config, check_dataset=False)
        baseline_config = load_config(args.baselines, validate=False)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    privacy_class = get_spec(config).name
    missing = [method for method in methods if method not in baseline_config]
    if missing:
        print(f"error: {args.baselines} has no section for: {', '.join(missing)}", file=sys.stderr)
        return 1
    sections = {method: _fill_privacy_class(baseline_config[method] or {}, privacy_class)
                for method in methods}

    print("\n" + "=" * 66)
    print("VeilLens — baselines")
    print("=" * 66)
    print(f"  methods       : {', '.join(methods)}")
    print(f"  privacy class : {privacy_class}")
    print(f"  dataset       : {config['dataset']['dataset_json_file']}")
    print(f"  attacker      : {config['evaluation']['model']}")
    print(f"  output        : {Path(config['output']['base_dir']) / 'baselines'}")
    if args.limit is not None:
        print(f"  limit         : first {args.limit} images")
    print("=" * 66 + "\n")

    problems = _check(config, methods, sections)
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
        print("\nRe-run with --dry-run after fixing, or see docs/BASELINES.md.", file=sys.stderr)
        return 1

    from veillens.baselines.runner import run_baseline

    failed: List[str] = []
    for method in methods:
        try:
            asyncio.run(run_baseline(
                config, method, sections[method],
                limit=args.limit, resume=not args.no_resume, evaluate=not args.no_eval,
            ))
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130
        except (RuntimeError, ValueError, OSError) as exc:
            # One method failing must not throw away the ones that already succeeded, but it
            # must not be reported as success either.
            print(f"\n{method} failed: {exc}", file=sys.stderr)
            failed.append(method)

    if failed:
        print(f"\n{len(failed)} of {len(methods)} method(s) failed: {', '.join(failed)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
