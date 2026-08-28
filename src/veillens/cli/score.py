#!/usr/bin/env python3
"""
Score a completed run, offline.

Reads the JSON a run already wrote and reports how much the protection changed what an
adversary could infer. No API calls, no GPU, no model weights — so this works on a laptop
against results produced on a cluster.

    veillens-score runs/geolocation/test_iter3
    veillens-score runs/compositional/test_large --setting compositional
    veillens-score runs/geolocation/test_iter3 --thresholds 0.5,1,5,10,25

For geolocation the metric is the share of images an adversary localises within N km
(``@0.5 km`` is roughly city-block precision). For compositional privacy it is the mean
model-assigned privacy-risk score, before and after. For social relationship it is the
adversary's pair-classification accuracy, six-way and three-way, before and after.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_THRESHOLDS_KM = (0.5, 1.0, 5.0, 10.0, 25.0, 100.0)


def _load_evaluation(run_dir: Path) -> Dict[str, Any]:
    candidates = [
        run_dir / "evaluation_results" / "evaluation_results.json",
        run_dir / "evaluation_results.json",
        run_dir,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(
        f"No evaluation_results.json under {run_dir}.\n"
        "Pass the run directory (the one containing evaluation_results/), or the JSON itself."
    )


def _split_before_after(data: Dict[str, Any], key: str) -> Tuple[List[float], List[float]]:
    """
    Pull the unprotected and protected value for each image.

    Every image records a list of `feature_results`; the entry with
    ``feature_removed == "none"`` is the original, and the last other entry is the final
    protected state.
    """
    before: List[float] = []
    after: List[float] = []
    for entry in data.values():
        results = entry.get("feature_results") or []
        for item in results:
            if item.get("feature_removed") == "none" and key in item:
                before.append(float(item[key]))
                break
        for item in reversed(results):
            if item.get("feature_removed") != "none" and key in item:
                after.append(float(item[key]))
                break
    return before, after


def _pct_within(values: List[float], threshold: float) -> float:
    if not values:
        return float("nan")
    return 100.0 * sum(1 for v in values if v < threshold) / len(values)


def _report_geolocation(data: Dict[str, Any], thresholds: Tuple[float, ...]) -> int:
    before, after = _split_before_after(data, "error_km")
    if not before:
        print("error: no 'error_km' entries found — is this a geolocation run?", file=sys.stderr)
        return 1

    print(f"\nGeolocation — {len(before)} images\n")
    print(f"  {'localised within':>18}   {'original':>9}   {'protected':>9}   {'change':>8}")
    print("  " + "-" * 54)
    for t in thresholds:
        b, a = _pct_within(before, t), _pct_within(after, t)
        label = f"{t:g} km"
        print(f"  {label:>18}   {b:8.1f}%   {a:8.1f}%   {a - b:+7.1f}")
    print("\n  median error   original {:>9.1f} km   protected {:>9.1f} km".format(
        statistics.median(before), statistics.median(after)))
    print("  mean error     original {:>9.1f} km   protected {:>9.1f} km".format(
        statistics.mean(before), statistics.mean(after)))
    return 0


def _report_compositional(data: Dict[str, Any]) -> int:
    before: List[float] = []
    after: List[float] = []
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        results = entry.get("feature_results") or []
        for item in results:
            score = (item.get("gpt_response") or {}).get("score")
            if score is None:
                continue
            if item.get("feature_removed") == "none":
                before.append(float(score))
                break
        for item in reversed(results):
            score = (item.get("gpt_response") or {}).get("score")
            if score is not None and item.get("feature_removed") != "none":
                after.append(float(score))
                break

    if not before:
        print("error: no privacy scores found — is this a compositional run?", file=sys.stderr)
        return 1

    mean_before, mean_after = statistics.mean(before), statistics.mean(after)
    reduction = 100.0 * (mean_before - mean_after) / mean_before if mean_before else float("nan")
    print(f"\nCompositional privacy — {len(before)} images\n")
    print(f"  mean risk score, original  : {mean_before:.4f}")
    print(f"  mean risk score, protected : {mean_after:.4f}")
    print(f"  relative reduction         : {reduction:.1f}%")
    return 0


def _relationship_totals(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-score every image's pairs from the responses the run recorded.

    Deliberately recomputed rather than read off ``error_relation``: the per-image error is one
    number over one label space, and the reported table needs four (six-way and three-way, over
    two denominators). Everything needed is in ``gpt_response.predictions``, which the spec writes
    in normalised pair-key form for exactly this reason.
    """
    from veillens.metrics.relationship import pair_accuracy

    totals = {
        "images": 0,
        "pairs": 0,
        "before": {"six": [0, 0], "three": [0, 0], "answered": 0},
        "after": {"six": [0, 0], "three": [0, 0], "answered": 0},
    }

    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        truth = entry.get("ground_truth_relations")
        if not truth:
            continue
        results = entry.get("feature_results") or []

        readings = {}
        for item in results:
            if item.get("feature_removed") == "none":
                readings.setdefault("before", item)
        for item in reversed(results):
            if item.get("feature_removed") != "none":
                readings.setdefault("after", item)
                break
        if "before" not in readings or "after" not in readings:
            continue

        totals["images"] += 1
        totals["pairs"] += len(truth)
        for when, item in readings.items():
            predictions = (item.get("gpt_response") or {}).get("predictions") or {}
            six = pair_accuracy(truth, predictions)
            three = pair_accuracy(truth, predictions, three_way=True)
            totals[when]["six"][0] += six.correct
            totals[when]["six"][1] += six.total
            totals[when]["three"][0] += three.correct
            totals[when]["three"][1] += three.total
            totals[when]["answered"] += six.answered

    return totals


def _report_social_relation(data: Dict[str, Any]) -> int:
    totals = _relationship_totals(data)
    if not totals["images"]:
        print("error: no relationship predictions found — is this a social-relationship run?",
              file=sys.stderr)
        return 1

    def accuracy(correct: int, denominator: int) -> float:
        return 100.0 * correct / denominator if denominator else float("nan")

    def cell(value: float) -> str:
        # A reduction against a zero baseline is undefined, not zero, and printing "nan%" invites
        # it to be read as a number. That happens whenever the adversary was already wrong on
        # every raw pair -- rare at scale, routine on a handful of images.
        return "      n/a" if value != value else f"{value:9.1f}%"

    print(f"\nSocial relationship — {totals['images']} images, {totals['pairs']} annotated pairs\n")

    # Two denominators, both reported. Over every annotated pair, a pair the adversary declined to
    # answer counts against it, which is the number the defense is entitled to claim. Over
    # answered pairs only, it does not -- that isolates how well the adversary does when it does
    # commit. They diverge exactly when the protected image stops the adversary answering at all,
    # which is a real effect but not the same claim, so neither is reported alone.
    for label, denominator_of in (
        ("all annotated pairs", lambda side, space: totals[side][space][1]),
        ("answered pairs only", lambda side, space: totals[side]["answered"]),
    ):
        print(f"  over {label}")
        print(f"    {'':10}{'original':>10}{'protected':>11}{'reduction':>11}")
        print("    " + "-" * 42)
        for space, name in (("six", "6-way"), ("three", "3-way")):
            before = accuracy(totals["before"][space][0], denominator_of("before", space))
            after = accuracy(totals["after"][space][0], denominator_of("after", space))
            reduction = 100.0 * (before - after) / before if before else float("nan")
            print(f"    {name:10}{cell(before)}{cell(after)}{cell(reduction)}")
        print()

    print(f"  pairs answered : original {totals['before']['answered']}, "
          f"protected {totals['after']['answered']}, of {totals['pairs']}")
    if totals["before"]["six"][0] == 0:
        print("  note           : the adversary recovered no pair on the *unprotected* images "
              "either,\n                   so there is no baseline to reduce and the reduction "
              "column is undefined.")
    if totals["after"]["answered"] < totals["pairs"]:
        print("  note           : the adversary left some pairs unanswered on the protected "
              "images.\n                   The two denominators above differ by exactly those "
              "pairs.")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="veillens-score",
        description="Score a completed run from its JSON output. No API keys or GPU needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  veillens-score runs/geolocation/test_iter3
  veillens-score examples/results/compositional
""",
    )
    parser.add_argument("run_dir", help="run directory, or an evaluation_results.json")
    parser.add_argument("--setting",
                        choices=["geolocation", "compositional", "social_relation", "auto"],
                        default="auto", help="which metric to report (default: sniff it)")
    parser.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS_KM),
                        help="comma-separated km thresholds for geolocation")
    args = parser.parse_args(argv)

    try:
        data = _load_evaluation(Path(args.run_dir))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if isinstance(data, dict) and "summary" in data and "results" in data:
        summary = data["summary"]
        print("\nCompositional privacy — summary block\n")
        for key, value in summary.items():
            print(f"  {key:26s}: {value}")
        return 0

    setting = args.setting
    if setting == "auto":
        # Reuses `veillens.verify.report.sniff_privacy_class`, which reads the same ERROR_KEYS
        # table, so veillens-score and veillens-verify can never disagree about what a results
        # file is -- and a fourth setting only has to be added in one place.
        from veillens.verify.report import sniff_privacy_class
        setting = sniff_privacy_class(data)

    if setting == "geolocation":
        thresholds = tuple(float(t) for t in args.thresholds.split(","))
        return _report_geolocation(data, thresholds)
    if setting == "social_relation":
        return _report_social_relation(data)
    return _report_compositional(data)


if __name__ == "__main__":
    raise SystemExit(main())
