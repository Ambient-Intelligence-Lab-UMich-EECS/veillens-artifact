#!/usr/bin/env python3
"""
Verify a completed run against the granularity it targeted, offline.

``veillens-score`` reports a distribution: how far off the attacker was. This reports a verdict:
whether the cue granularities the run promised to remove are actually gone. Two criteria, because
they answer different questions and can disagree —

    metric    the attacker's estimate is at least `threshold` km from the truth, where the
              threshold is derived from experiment.mask_level rather than set by hand
    residual  the attacker's own report names no surviving cue at a protected granularity

No API calls, no GPU, no model weights, and no config file: a run records the granularity it
targeted in its own results, so archived runs verify too.

    veillens-verify runs/geolocation/test_iter3
    veillens-verify examples/results/geolocation
    veillens-verify examples/results/compositional
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from veillens.verify.granularity import CLUE_LEVEL_TO_RUNG, COMPOSITIONAL, GEOLOCATION
from veillens.verify.report import (
    find_response_dirs,
    infer_mask_level,
    load_responses,
    sniff_privacy_class,
    verify_run,
)


def _load_evaluation(run_dir: Path) -> Dict[str, Any]:
    """Same candidate order as ``veillens-score``, so both accept the same argument."""
    candidates = [
        run_dir / "evaluation_results" / "evaluation_results.json",
        run_dir / "evaluation_results.json",
        run_dir,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text())
    hint = ""
    if (run_dir / "index.csv").is_file():
        # The archived results/ trees ship a per-run index.csv and the attacker's raw responses,
        # but not the evaluation_results.json those numbers were derived from. Deliberately not
        # parsed here: every archive has a different column set, and guessing which column is the
        # protected error is exactly the kind of drift this tool exists to catch. Point at the
        # pairing instead.
        hint = (
            f"\n\n{run_dir} is an archived run: it has index.csv and the raw responses, but no "
            "evaluation_results.json, so there is no per-image error to verify against.\n"
            "Read it as a table instead, or point this at a run directory that has one and name "
            "these responses:\n\n"
            "    veillens-verify <run-with-evaluation_results.json> \\\n"
            f"        --responses {run_dir}/evaluation/<model>/responses\n"
        )
    raise FileNotFoundError(
        f"No evaluation_results.json under {run_dir}.\n"
        "Pass the run directory (the one containing evaluation_results/), or the JSON itself."
        + hint
    )


def _parse_mask_level(raw: str, privacy_class: str) -> Any:
    if privacy_class == COMPOSITIONAL:
        return int(raw)
    levels = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [level for level in levels if level not in CLUE_LEVEL_TO_RUNG]
    if unknown:
        raise ValueError(
            f"unknown clue level(s): {', '.join(unknown)}. "
            f"Expected any of: {', '.join(CLUE_LEVEL_TO_RUNG)}"
        )
    return levels


def _pick_response_dir(run_dir: Path, override: Optional[str]) -> Optional[Path]:
    """
    Locate the attacker's raw responses, or return None having said why.

    Ambiguity is refused rather than guessed: a run with several evaluated variants (one per
    attacker model, or one per defence baseline) would otherwise be verified against whichever
    directory happened to sort first.
    """
    if override:
        chosen = Path(override)
        if not chosen.is_dir():
            print(f"error: --responses {chosen} is not a directory", file=sys.stderr)
            raise SystemExit(1)
        return chosen

    found: List[Path] = find_response_dirs(run_dir)
    if not found:
        return None
    if len(found) == 1:
        return found[0]

    print(f"note: {len(found)} response directories under {run_dir}; "
          f"pass --responses to choose one:", file=sys.stderr)
    for candidate in found:
        count = len(list(candidate.glob("*.json")))
        print(f"        --responses {candidate}   ({count} files)", file=sys.stderr)
    print("      the residual criterion will not be computed until one is chosen.",
          file=sys.stderr)
    return None


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="veillens-verify",
        description="Verify a run against the cue granularity it targeted. "
                    "No API keys, GPU or weights needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  veillens-verify runs/geolocation/test_iter3
  veillens-verify results/geolocation_100
  veillens-verify results/geolocation_100 --json /tmp/verification.json
""",
    )
    parser.add_argument("run_dir", help="run directory, or an evaluation_results.json")
    parser.add_argument("--setting", choices=["geolocation", "compositional", "social_relation", "auto"],
                        default="auto", help="privacy class (default: sniff it from the results)")
    parser.add_argument("--mask-level", default=None,
                        help="override the protected granularity, e.g. "
                             "'street level clue,city level clue' or, for compositional, '2'. "
                             "By default it is recovered from the run's own results.")
    parser.add_argument("--responses", default=None,
                        help="directory of raw attacker responses, if it cannot be found "
                             "automatically or if the run has more than one")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also write the full report to this path")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    try:
        data = _load_evaluation(run_dir)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, dict) or not data:
        print("error: evaluation_results.json is empty or not a mapping of image ids",
              file=sys.stderr)
        return 1

    if "summary" in data and "results" in data:
        # A post-hoc aggregate, not the loop's own per-image record. It carries a final score but
        # not `feature_removed`, so the granularity a run targeted cannot be recovered from it and
        # there is nothing to verify against. `veillens-score` accepts this shape; verification
        # needs the run's own file.
        print("error: this is a summary aggregate, not a per-image run record — it does not say "
              "which granularity was protected.\n"
              "       Verify the run's own evaluation_results.json instead (in this repository, "
              "the compositional one is\n"
              "       examples/results/compositional/evaluation_results/evaluation_results.json).",
              file=sys.stderr)
        return 1

    privacy_class = args.setting if args.setting != "auto" else sniff_privacy_class(data)

    if args.mask_level is not None:
        try:
            mask_level = _parse_mask_level(args.mask_level, privacy_class)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        mask_level = infer_mask_level(data, privacy_class)
        if privacy_class == GEOLOCATION and not mask_level:
            print("error: could not recover mask_level from the results "
                  "(no 'feature_removed' label names a clue granularity). "
                  "Pass --mask-level.", file=sys.stderr)
            return 1
        if privacy_class == COMPOSITIONAL and mask_level is None:
            print("error: could not recover mask_level from the results. Pass --mask-level.",
                  file=sys.stderr)
            return 1

    search_root = run_dir if run_dir.is_dir() else run_dir.parent
    response_dir = _pick_response_dir(search_root, args.responses)
    responses = load_responses(response_dir) if response_dir else None

    report = verify_run(
        data,
        responses,
        privacy_class=privacy_class,
        mask_level=mask_level,
        residual_source=str(response_dir) if response_dir else None,
    )
    print(report.render())

    if args.json_out:
        written = report.write(Path(args.json_out))
        print(f"  wrote {written}\n")

    # Exit 0 whether or not images failed: a failing image is a measurement, not a tool error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
