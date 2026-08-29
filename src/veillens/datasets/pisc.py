"""
Build a social-relationship manifest from a PISC checkout.

python -m veillens.datasets.pisc \\
    --pisc-root $VEILLENS_DATA/pisc \\
    --num-images 500 --seed 0 \\
    --annotated-dir $VEILLENS_DATA/pisc_annotated \\
    --output manifests/social_relation_500.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from veillens.datasets.schema import SCHEMA_VERSION
from veillens.datasets.subjects import draw_subject_boxes
from veillens.metrics.relationship import LABELS

logger = logging.getLogger(__name__)

#: Relative to ``--pisc-root``. The split files are flat JSON arrays of unpadded id strings.
DEFAULT_SPLIT = "relationship_split/relation_testidx.json"


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"PISC file not found: {path}\n"
            "Point --pisc-root at a PISC checkout; see docs/DATA.md for the layout."
        )
    return json.loads(path.read_text())


def load_pisc(pisc_root: Path, split: str) -> Tuple[List[str], Dict, Dict]:
    """
    Read one split index, the relationship labels, and the box annotations.

    Returns:
        ``(split_ids, relationships, annotations)``. ``annotations`` is re-keyed to the same
        unpadded-string ids the other two files use, so nothing downstream has to remember that
        one of the three stores its id as an int.
    """
    split_ids = _load_json(pisc_root / split)
    relationships = _load_json(pisc_root / "relationship.json")
    raw_annotations = _load_json(pisc_root / "annotation_image_info.json")

    annotations = {str(entry["id"]): entry for entry in raw_annotations}
    return [str(i) for i in split_ids], relationships, annotations


def _pair_is_usable(pair_key: str, subject_count: int) -> bool:
    """A pair naming a subject with no box cannot be shown to an attacker, let alone protected."""
    parts = pair_key.split()
    if len(parts) != 2:
        return False
    try:
        first, second = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return 1 <= first <= subject_count and 1 <= second <= subject_count and first != second


def build_items(pisc_root: Path,
                annotated_dir: Path,
                dataset_name: str,
                split: str = DEFAULT_SPLIT,
                num_images: Optional[int] = None,
                seed: int = 0,
                render: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Select images from *split*, render their overlays, and return the manifest items.

    Sampling is a seeded shuffle over the split rather than a prefix of it, because the split file
    is not in a meaningful order and taking the first N would silently correlate the subset with
    whatever order the dataset authors happened to write.

    Returns:
        ``(items, skipped)`` where *skipped* counts each reason an image was dropped. The counts
        are returned rather than only logged: "500 images" means something different when 200 of
        the candidates had no image file on disk, and that has to be visible.
    """
    split_ids, relationships, annotations = load_pisc(pisc_root, split)
    image_dir = pisc_root / "image"

    ordered = list(split_ids)
    random.Random(seed).shuffle(ordered)

    skipped = {"no_labels": 0, "no_boxes": 0, "too_few_subjects": 0,
               "no_usable_pair": 0, "missing_image": 0, "unreadable_image": 0}
    items: List[Dict[str, Any]] = []

    if render:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    for image_id in ordered:
        if num_images is not None and len(items) >= num_images:
            break

        pairs = relationships.get(image_id)
        if not pairs:
            skipped["no_labels"] += 1
            continue

        annotation = annotations.get(image_id)
        if annotation is None:
            skipped["no_boxes"] += 1
            continue

        boxes = annotation.get("bbox") or []
        if len(boxes) < 2:
            # One person cannot form a pair, so there is nothing to infer and nothing to protect.
            skipped["too_few_subjects"] += 1
            continue

        usable = {key: int(value) for key, value in pairs.items()
                  if _pair_is_usable(key, len(boxes))}
        if not usable:
            skipped["no_usable_pair"] += 1
            continue

        source = image_dir / f"{int(image_id):05d}.jpg"
        if not source.is_file():
            skipped["missing_image"] += 1
            continue

        destination = annotated_dir / f"{image_id}.jpg"
        if render and not _render(source, destination, boxes):
            skipped["unreadable_image"] += 1
            continue

        items.append({
            "image_id": image_id,
            "dataset": dataset_name,
            "relpath": destination.name,
            "ground_truth_relations": usable,
            # Subject k is bbox[k-1]; storing the mapping explicitly means nothing downstream has
            # to re-derive it from a list position.
            "subjects": {str(index + 1): [int(v) for v in box]
                         for index, box in enumerate(boxes)},
            "image_size": [int(annotation.get("imgW", 0)), int(annotation.get("imgH", 0))],
        })

    return items, skipped


def _render(source: Path, destination: Path, boxes: List[List[float]]) -> bool:
    """Write the box overlay for one image. False if the source could not be read."""
    import cv2

    image = cv2.imread(str(source))
    if image is None:
        logger.warning("Could not read %s", source)
        return False

    annotated = draw_subject_boxes(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes)
    cv2.imwrite(str(destination), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return True


def _label_histogram(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {name: 0 for name in LABELS.values()}
    for item in items:
        for code in item["ground_truth_relations"].values():
            name = LABELS.get(int(code))
            if name:
                counts[name] += 1
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m veillens.datasets.pisc",
        description="Build a social-relationship manifest, with box overlays, from PISC.",
    )
    parser.add_argument("--pisc-root", required=True, type=Path,
                        help="PISC checkout: relationship.json, annotation_image_info.json, image/")
    parser.add_argument("--annotated-dir", required=True, type=Path,
                        help="where the box-overlay copies are written")
    parser.add_argument("--output", required=True, type=Path,
                        help="manifest to write")
    parser.add_argument("--dataset-name", default=None,
                        help="the manifest's 'dataset' field, joined against $VEILLENS_DATA "
                             "(default: the annotated directory's name)")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help=f"split index, relative to --pisc-root (default: {DEFAULT_SPLIT})")
    parser.add_argument("--num-images", type=int, default=None,
                        help="how many images to keep (default: every usable one)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the sampling shuffle")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s: %(message)s")

    dataset_name = args.dataset_name or args.annotated_dir.name

    try:
        items, skipped = build_items(
            pisc_root=args.pisc_root,
            annotated_dir=args.annotated_dir,
            dataset_name=dataset_name,
            split=args.split,
            num_images=args.num_images,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not items:
        print("error: no usable images found. Skipped: "
              + ", ".join(f"{k}={v}" for k, v in skipped.items() if v), file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # `load_manifest` reads only "items", so the provenance block costs nothing at load time and
    # travels with the file. It matters because a subset drawn while the image archive is only
    # partly unpacked is not a uniform sample of its split, and the skip counts are the only
    # record of that -- without them a later reader cannot tell one case from the other.
    args.output.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "source": "PISC",
            "split": args.split,
            "seed": args.seed,
            "requested": args.num_images,
            "selected": len(items),
            "pairs": sum(len(i["ground_truth_relations"]) for i in items),
            "skipped": {reason: count for reason, count in skipped.items() if count},
            "labels": _label_histogram(items),
        },
        "items": items,
    }, indent=2) + "\n")

    pairs = sum(len(item["ground_truth_relations"]) for item in items)
    print(f"\nwrote {args.output}")
    print(f"  images        : {len(items)}")
    print(f"  annotated     : {args.annotated_dir}  (dataset: {dataset_name})")
    print(f"  pairs         : {pairs}")
    print("  labels        : "
          + ", ".join(f"{name} {count}" for name, count in _label_histogram(items).items()))
    dropped = {reason: count for reason, count in skipped.items() if count}
    if dropped:
        print("  skipped       : " + ", ".join(f"{k}={v}" for k, v in dropped.items()))
        if skipped["missing_image"]:
            print(f"  note          : {skipped['missing_image']} candidates had no image file. "
                  "If that is most of the split, the image archive is only partly unpacked "
                  "(see docs/DATA.md).")
    if args.num_images is not None and len(items) < args.num_images:
        print(f"  warning       : asked for {args.num_images}, the split yielded {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
