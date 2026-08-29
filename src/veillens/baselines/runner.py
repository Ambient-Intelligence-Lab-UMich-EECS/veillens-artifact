"""
Everything that surrounds a baseline: manifest in, score table out.

A baseline only knows how to protect one image. This module does the rest, and does it the same
way for all five, so that the only difference between two rows of the baseline table is the
transform itself:

1. resolve the manifest the pipeline config names;
2. apply the method to every image, writing `defended/<image_id>.png` and per-image stats;
3. run **the attacker** over the originals and over the protected images;
4. write `evaluation_results/evaluation_results.json` in exactly the shape the protection loop
   writes, so `veillens-score` reads a baseline run with no flags and no special case.

Attacker predictions on unmodified images are cached in `baselines/_attacker_cache/`, shared
across methods, because five baselines over one manifest would otherwise pay for the same
unprotected reading five times.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from veillens.baselines import build
from veillens.datasets.schema import load_manifest
from veillens.privacy.spec import get_spec
from veillens.reason.oracle import PrivacyOracle
from veillens.vlm.registry import client_from_config

logger = logging.getLogger(__name__)

__all__ = ["run_baseline", "baseline_output_dir"]


def baseline_output_dir(config: Dict[str, Any], method: str) -> Path:
    """Where a method's output lands: `<output.base_dir>/baselines/<method>/`."""
    return Path(config["output"]["base_dir"]) / "baselines" / method


def _changed_pixels(original: Path, protected: Path) -> Tuple[int, int]:
    """
    How many pixels the method actually moved, and how many there are.

    Reported into the same `mask_pixels` / `mask_percentage` fields the loop fills from its
    inpainting masks, which makes the "how much of the frame did you touch" column comparable
    across methods that have no mask at all.
    """
    with Image.open(original) as handle:
        before = np.asarray(handle.convert("RGB"))
    with Image.open(protected) as handle:
        after = np.asarray(handle.convert("RGB"))
    if before.shape != after.shape:
        return 0, int(before.shape[0] * before.shape[1])
    differing = np.any(before != after, axis=2)
    return int(np.count_nonzero(differing)), int(differing.size)


async def _protect_images(baseline, items: List[Dict[str, Any]], dirs: Dict[str, Path],
                          resume: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Apply the method to every item. Returns (protected, failures)."""
    semaphore = asyncio.Semaphore(max(1, baseline.workers))
    protected: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    async def one(item: Dict[str, Any]) -> None:
        image_id = item["image_id"]
        destination = dirs["defended"] / f"{image_id}.png"
        if resume and destination.is_file():
            logger.info("skipping %s (already protected)", image_id)
            protected.append({"item": item, "path": destination, "stats": {"resumed": True}})
            return

        async with semaphore:
            started = time.time()
            try:
                image, stats = await asyncio.to_thread(
                    baseline.apply, item["image_path"], image_id
                )
            except Exception as exc:  # noqa: BLE001 - one bad image must not end the run
                logger.error("%s failed on %s: %s", baseline.name, image_id, exc)
                failures.append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
                return

        image.save(destination)
        stats = dict(stats)
        stats["elapsed_s"] = round(time.time() - started, 2)
        (dirs["metadata"] / f"{image_id}.json").write_text(json.dumps(stats, indent=2))
        print(f"  [{baseline.name}] {image_id}  ({stats['elapsed_s']}s)")
        protected.append({"item": item, "path": destination, "stats": stats})

    await asyncio.gather(*(one(item) for item in items))
    # asyncio.gather preserves argument order but the appends above do not, so re-sort to keep
    # the results file and the console log in manifest order.
    protected.sort(key=lambda entry: str(entry["item"]["image_id"]))
    return protected, failures


async def _predict(oracle: PrivacyOracle, schema_type: str, image_path: Path,
                   cache: Path) -> Optional[Dict[str, Any]]:
    """One attacker reading, cached on disk so a re-run costs nothing."""
    if cache.is_file():
        return json.loads(cache.read_text())
    result = await oracle.get_geolocation_prediction(
        str(image_path), schema_type=schema_type, role="eval"
    )
    if result is not None:
        cache.write_text(json.dumps(result, indent=2))
    return result


async def run_baseline(config: Dict[str, Any], method: str, params: Dict[str, Any],
                       limit: Optional[int] = None, resume: bool = True,
                       evaluate: bool = True) -> Path:
    """
    Run one baseline end to end and return its output directory.

    Args:
        config: a loaded pipeline config — the same file `veillens-protect` takes. The dataset,
            the attacker and the output root all come from it, which is what makes a baseline
            row comparable to the VeilLens row.
        method: registry name, e.g. ``"rand-noise"``.
        params: the method's section of `configs/baselines.yaml`.
        limit: process only the first N manifest entries. For smoke tests.
        resume: skip images that already have a protected output.
        evaluate: run the attacker pass. Off, this writes images and stops.

    Raises:
        RuntimeError: every image failed, so there is nothing to evaluate. A run that protected
            nothing must not exit reporting a score table.
    """
    spec = get_spec(config)
    items = load_manifest(Path(config["dataset"]["dataset_json_file"]))
    if limit is not None:
        items = items[:limit]

    root = baseline_output_dir(config, method)
    dirs = {
        "root": root,
        "defended": root / "defended",
        "metadata": root / "metadata",
        "attacker_cache": root / "attacker_cache",
        "evaluation_results": root / "evaluation_results",
        # Shared across methods: the unprotected reading does not depend on the method.
        "original_cache": root.parent / "_attacker_cache",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    baseline = build(method, params)
    print(f"\n[baseline] {method}: {baseline.summary}")
    print(f"[baseline] {len(items)} images -> {root}")

    baseline.prepare(items)
    try:
        protected, failures = await _protect_images(baseline, items, dirs, resume)
    finally:
        baseline.finish()

    if not protected:
        raise RuntimeError(
            f"{method} protected 0 of {len(items)} images, so there is nothing to evaluate.\n"
            "The per-image errors are in the log above at ERROR level."
        )
    if failures:
        print(f"\n[baseline] {len(failures)} of {len(items)} images failed:")
        for failure in failures[:5]:
            print(f"  - {failure['image_id']}: {failure['error']}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    summary = {
        "method": method,
        "params": params,
        "images_requested": len(items),
        "images_protected": len(protected),
        "failures": failures,
    }

    if not evaluate:
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[baseline] wrote {len(protected)} protected images; evaluation skipped")
        return root

    # Which schema the attacker answers under is the spec's to say -- the compositional setting
    # runs its final read through the round-1 schema, which already exposes the privacy score.
    # Reading it from the spec rather than restating the rule is what keeps a baseline's numbers
    # comparable with the loop's: there is now one place the rule can be wrong.
    schema_type = spec.eval_schema_type

    evaluation_section = config["evaluation"]
    client = client_from_config(evaluation_section)
    oracle = PrivacyOracle(
        model=evaluation_section["model"],
        concurrent_requests=evaluation_section["concurrent_requests"],
        debug_dir=root / "debug_cache",
        spec=spec,
        reason_client=client,
        eval_client=client,
    )

    print(f"\n[baseline] attacker pass: {evaluation_section['model']} over "
          f"{len(protected)} originals and {len(protected)} protected images")

    async def evaluate_one(entry: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        item = entry["item"]
        image_id = item["image_id"]
        before, after = await asyncio.gather(
            _predict(oracle, schema_type, Path(item["image_path"]),
                     dirs["original_cache"] / f"{image_id}.json"),
            _predict(oracle, schema_type, entry["path"],
                     dirs["attacker_cache"] / f"{image_id}.json"),
        )
        if before is None or after is None:
            logger.error("attacker returned nothing for %s; dropping it from the table", image_id)
            return None

        try:
            ground_truth = spec.normalize_ground_truth(item[spec.ground_truth_field])
            error_before = spec.compute_error(ground_truth, before)
            error_after = spec.compute_error(ground_truth, after)
        except (KeyError, TypeError, ValueError) as exc:
            # A response that parsed as JSON but does not carry the field the spec scores on.
            # Drop the image rather than lose the whole table to one malformed answer.
            logger.error("cannot score %s: %s", image_id, exc)
            return None
        changed, total = _changed_pixels(Path(item["image_path"]), entry["path"])

        return image_id, {
            spec.gt_output_key: spec.serialize_ground_truth(ground_truth),
            "feature_results": [
                {
                    "feature_removed": "none",
                    spec.error_key: error_before,
                    "gpt_response": spec.extract_prediction_summary(before),
                    spec.error_increase_key: 0.0,
                    "mask_pixels": 0,
                    "mask_percentage": 0.0,
                },
                {
                    "feature_removed": method,
                    spec.error_key: error_after,
                    "gpt_response": spec.extract_prediction_summary(after),
                    spec.error_increase_key: error_after - error_before,
                    "mask_pixels": changed,
                    "mask_percentage": float(changed / total * 100) if total else 0.0,
                },
            ],
        }

    evaluated = await asyncio.gather(*(evaluate_one(entry) for entry in protected))
    results = {image_id: payload for image_id, payload in filter(None, evaluated)}

    if not results:
        raise RuntimeError(
            f"{method}: the attacker returned nothing for any of {len(protected)} images, so no "
            "score table can be computed. Check the API key and the errors logged above."
        )

    destination = dirs["evaluation_results"] / "evaluation_results.json"
    destination.write_text(json.dumps(results, indent=2))

    summary["images_evaluated"] = len(results)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[baseline] wrote {destination}")
    print(f"[baseline] score it with:  veillens-score {root}")
    return root
