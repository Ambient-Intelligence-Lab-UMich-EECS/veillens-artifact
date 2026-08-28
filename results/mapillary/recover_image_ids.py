#!/usr/bin/env python3
"""Re-derive which image each response in the Mapillary evaluation run belongs to.

The source run stores its responses as a positional array with no image id:

    <run>/raw/<provider>/<model>/results.json
        responses[i]             = {"reasoning": ..., "location_estimate": {...}}
        prediction_errors_km[i]  = great-circle error of responses[i]
        failures                 = the image ids whose API call failed

and the manifest that fixed the order, ``selected_eval.json``, no longer exists. The
mapping in this folder was recovered from the two things that do survive: the guessed
coordinate in each response, and the error recorded beside it.

For response ``i`` the error pins the ground truth: the run used a haversine with
R = 6371.0 km, so the image is whichever candidate ``g`` satisfies

    | haversine(g, responses[i].location_estimate) - prediction_errors_km[i] | < 1e-9

Over the 217 candidates in ``mapillary/mega/selected.json`` that is unique for 152 of the
158 slots. The remaining 6 follow by force: the resolved slots are strictly increasing in
selected.json order, and each unresolved run of k slots sits in a window of exactly k
candidates, so there is one assignment consistent with the ordering.

The result is checked against evidence that was never used to build it: the ``failures``
list names the images whose calls failed, and those ids land exactly on the slots whose
response is null -- 2/2 for Haiku, 1/1 for Gemini 3 Flash.

Usage:  python recover_image_ids.py [--pipeline2 /path/to/pipeline2]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EARTH_RADIUS_KM = 6371.0  # pipeline2/utils/distanceCalculator.py
RUN = "result/mega_city_street/mega_city_street_eval_20260208_204436/raw"
MODELS = ("claude/claude-haiku-4-5", "gemini/gemini-3-flash-preview")
TOL_KM = 1e-9


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def match_by_error(results: dict, truth: dict[str, tuple[float, float]]) -> list[str | None]:
    """Slot -> image id, for slots where exactly one ground truth reproduces the error."""
    out: list[str | None] = []
    for response, error in zip(results["responses"], results["prediction_errors_km"]):
        estimate = (response or {}).get("location_estimate") or {}
        try:
            guess = (float(estimate["latitude"]), float(estimate["longitude"]))
        except (KeyError, TypeError, ValueError):
            out.append(None)
            continue
        if error is None:
            out.append(None)
            continue
        hits = [i for i, g in truth.items() if abs(haversine(g, guess) - error) < TOL_KM]
        out.append(hits[0] if len(hits) == 1 else None)
    return out


def fill_forced(slots: list[str | None], order: list[str]) -> list[str | None]:
    """Fill runs of unknown slots whose candidate window admits exactly one ordering."""
    position = {image_id: i for i, image_id in enumerate(order)}
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(slots):
            if slots[i] is not None:
                i += 1
                continue
            j = i
            while j < len(slots) and slots[j] is None:
                j += 1
            low = position[slots[i - 1]] if i > 0 else -1
            high = position[slots[j]] if j < len(slots) else len(order)
            window = list(range(low + 1, high))
            if len(window) == j - i:
                for k, c in enumerate(window):
                    slots[i + k] = order[c]
                changed = True
            i = j
    return slots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline2", type=Path, default=Path("pipeline2"),
                    help="root of the pipeline2 tree holding mapillary/ and result/")
    ap.add_argument("--out", type=Path, help="write the recovered id list here as JSON")
    args = ap.parse_args()

    selected = json.loads((args.pipeline2 / "mapillary/mega/selected.json").read_text())
    order = [str(e["image_id"]) for e in selected]
    truth = {str(e["image_id"]): tuple(e["ground_truth_coords"]) for e in selected}

    sources = {m: json.loads((args.pipeline2 / RUN / m / "results.json").read_text())
               for m in MODELS}
    per_model = {m: match_by_error(d, truth) for m, d in sources.items()}

    merged: list[str | None] = []
    for slot in zip(*per_model.values()):
        found = {s for s in slot if s}
        assert len(found) <= 1, f"evaluators disagree on a slot: {found}"
        merged.append(next(iter(found)) if found else None)
    by_distance = sum(1 for s in merged if s)

    merged = fill_forced(merged, order)
    assert all(merged) and len(set(merged)) == len(merged), "recovery is incomplete"
    position = {image_id: i for i, image_id in enumerate(order)}
    ranks = [position[s] for s in merged]
    assert all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1)), "not monotone"

    print(f"{len(merged)} slots: {by_distance} matched on error, "
          f"{len(merged) - by_distance} forced by ordering")

    for model, data in sources.items():
        null_slots = [i for i, (r, e) in
                      enumerate(zip(data["responses"], data["prediction_errors_km"]))
                      if e is None or not r]
        recovered = sorted(merged[i] for i in null_slots)
        declared = sorted(data["failures"])
        assert recovered == declared, f"{model}: {recovered} != {declared}"
        print(f"  {model}: failures land on {recovered} -- matches the run's own list")

        wrong = sum(
            1 for i, (r, e) in enumerate(zip(data["responses"], data["prediction_errors_km"]))
            if e is not None
            and abs(haversine(truth[merged[i]],
                              (float(r["location_estimate"]["latitude"]),
                               float(r["location_estimate"]["longitude"]))) - e) >= TOL_KM
        )
        assert wrong == 0
        print(f"  {model}: every recorded error reproduces from the recovered ground truth")

    if args.out:
        args.out.write_text(json.dumps(merged, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
