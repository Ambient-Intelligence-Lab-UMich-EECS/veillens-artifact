"""
Privacy class specifications.

A `PrivacyClassSpec` bundles everything that differs between privacy classes — prompt and schema
paths, which cues count as in-scope for a given mask level, how round-2 context is phrased, and
what "error" means — so the loop can stay generic instead of branching on a class name.

This is the extension point of the whole system. Adding a privacy setting means writing one
spec instance and registering it in `veillens.privacy.registry`; nothing in `veillens.loop`
changes. See `docs/ARCHITECTURE.md` for how the loop consumes a spec.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from veillens.datasets.subjects import head_band
from veillens.metrics.geodesic import DistanceCalculator
from veillens.metrics.relationship import pair_accuracy
from veillens.paths import resolve as _resolve

_distance_calc = DistanceCalculator()


def _load_text(path_like: str) -> str:
    with open(_resolve(path_like), "r") as f:
        return f.read()


def _load_json(path_like: str) -> Dict:
    import json
    with open(_resolve(path_like), "r") as f:
        return json.load(f)


# ============================================================================
# Spec dataclass
# ============================================================================


@dataclass
class PrivacyClassSpec:
    """
    Bundles all privacy-class-specific behavior into a single object.

    All *_path fields are paths (with extension) relative to the project root,
    e.g. ``"prompts/geolocation/schema_original.json"``.
    """
    name: str
    anonymized_field: str
    privacy_field: str
    ground_truth_field: str
    gt_output_key: str
    error_key: str
    error_increase_key: str
    error_unit: str
    round1_schema_path: str
    round1_prompt_path: str
    round2_schema_path: str
    round2_prompt_path: str
    eval_schema_path: str
    eval_prompt_path: str

    feature_passes_mask_level: Callable[[Dict, Any], bool] = None  # type: ignore[assignment]
    format_round1_context: Callable[[Dict], str] = None  # type: ignore[assignment]
    build_round2_prompt: Callable[[str, str, str], str] = None  # type: ignore[assignment]
    compute_error: Callable[[Any, Dict], float] = None  # type: ignore[assignment]
    extract_prediction_summary: Callable[[Dict], Dict] = None  # type: ignore[assignment]
    normalize_ground_truth: Callable[[Any], Any] = None  # type: ignore[assignment]
    serialize_ground_truth: Callable[[Any], Any] = None  # type: ignore[assignment]
    format_mask_level_label: Callable[[Any], str] = None  # type: ignore[assignment]

    #: Which schema the attacker answers under. Geolocation has a dedicated eval schema; the
    #: compositional setting reuses its round-1 schema, which already exposes the risk score. This
    #: replaced three copies of `'eval' if spec.name == 'geolocation' else 'original'` scattered
    #: across the loop, the driver and the baseline runner.
    eval_schema_type: str = "original"

    #: Measure the "before" reading with the *attacker* on the unedited image, rather than reusing
    #: the defense oracle's round-1 response. Costs one extra call per image, and is what a setting
    #: needs when its headline number is a before/after comparison of the same adversary.
    eval_original: bool = False

    #: Cap on the input's long side, in pixels, or None to process at native resolution.
    #: Not a performance tweak: nothing in the edit path bounds resolution -- SDXL inpaints at
    #: whatever size it is handed -- so a multi-megapixel source asks for a single allocation
    #: large enough to exhaust a 48 GB GPU. A setting whose dataset ships full-frame camera
    #: originals has to say so here; one whose images are already bounded can leave it None.
    max_input_dim: Optional[int] = None

    #: Return SAM 2 box prompts for a feature, or None to fall through to Grounding DINO. Lets a
    #: setting whose regions the dataset already supplies skip open-set detection entirely.
    #: Signature: (feature, item, options) -> Optional[np.ndarray] of pixel xyxy boxes, where
    #: `item` is the manifest row and `options` is the config's `intervention:` section.
    feature_boxes: Callable[[Dict, Any, Dict], Any] = None  # type: ignore[assignment]

    #: Narrow a mask SAM 2 returned before it is inpainted, or None to use it as-is. Runs once
    #: per mask, in the seg worker, right after segmentation.
    #: Signature: (mask, feature, item, options) -> np.ndarray of the same shape.
    refine_mask: Callable[[Any, Dict, Any, Dict], Any] = None  # type: ignore[assignment]

    #: Per-image text appended to every prompt this setting sends. A setting that asks about
    #: specific subjects has to name them; the shipped settings ask about the whole image and
    #: leave this None, which reproduces the static-prompt behaviour exactly.
    format_item_context: Callable[[Any], Any] = None  # type: ignore[assignment]

    _schemas: Dict[str, Dict] = None  # type: ignore[assignment]
    _prompts: Dict[str, str] = None  # type: ignore[assignment]

    # -- schema / prompt accessors ------------------------------------------

    def get_schema(self, mode: str) -> Dict:
        if self._schemas is None:
            self._schemas = {}
        mapping = {"r1": self.round1_schema_path,
                   "r2": self.round2_schema_path,
                   "eval": self.eval_schema_path}
        path = mapping[mode]
        if path not in self._schemas:
            self._schemas[path] = _load_json(path)
        return self._schemas[path]

    def _get_prompt(self, path: str) -> str:
        if self._prompts is None:
            self._prompts = {}
        if path not in self._prompts:
            self._prompts[path] = _load_text(path)
        return self._prompts[path]

    def get_round1_prompt(self) -> str:
        return self._get_prompt(self.round1_prompt_path)

    def get_round2_prompt(self) -> str:
        return self._get_prompt(self.round2_prompt_path)

    def get_eval_prompt(self) -> str:
        return self._get_prompt(self.eval_prompt_path)

    def format_round2_prompt(self, r1_context: str, region_name: str) -> str:
        base = self.get_round2_prompt()
        return self.build_round2_prompt(r1_context, region_name, base)


# ============================================================================
# Geolocation behaviors
# ============================================================================


def _geo_passes(feature: Dict, mask_levels: List[str]) -> bool:
    return any(level in feature.get("properties", []) for level in mask_levels)


def _geo_format_r1_context(result: Dict) -> str:
    loc = result.get("location_estimate", {})
    return (
        f"{loc.get('region', 'Unknown')} "
        f"(lat: {loc.get('latitude', 0)}, lon: {loc.get('longitude', 0)})"
    )


def _geo_build_r2_prompt(r1_context: str, region_name: str, base_phase2_prompt: str) -> str:
    return base_phase2_prompt.format(predicted_location=r1_context, region_name=region_name)


def _geo_compute_error(ground_truth, prediction_result: Dict) -> float:
    loc = prediction_result["location_estimate"]
    return _distance_calc.haversine_distance(
        ground_truth, (loc["latitude"], loc["longitude"])
    )


def _geo_extract_summary(result: Dict) -> Dict:
    return result["location_estimate"]


def _geo_normalize_gt(raw):
    return tuple(raw)


def _geo_serialize_gt(value):
    return list(value)


def _geo_mask_level_label(mask_levels: List[str]) -> str:
    return ", ".join(mask_levels)


# ============================================================================
# Personal-info behaviors
# ============================================================================


def _pi_passes(feature: Dict, mask_level: int) -> bool:
    severity = feature.get("severity_level")
    if severity is None:
        return False
    return int(severity) <= int(mask_level)


def _pi_format_r1_context(result: Dict) -> str:
    score_obj = result.get("personal_info_privacy_score", {}) or {}
    score = score_obj.get("score", 0.0)
    confidence = score_obj.get("confidence", "Unknown")

    features = (result.get("critical_features", {}) or {}).get("features", []) or []
    cats = Counter()
    for feat in features:
        for attr in feat.get("attributes", []) or []:
            cats[attr.strip()] += 1
    top = ", ".join(name for name, _ in cats.most_common(3)) if cats else "none"

    return (
        f"Round 1 privacy score: {score} (confidence: {confidence}); "
        f"dominant categories: {top}"
    )


def _pi_build_r2_prompt(r1_context: str, region_name: str, base_prompt: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"CONTEXT FROM ROUND 1: {r1_context}\n"
        f"You are now analyzing a regional crop labeled '{region_name}'. "
        f"Re-evaluate this region independently and report any privacy-relevant elements visible in the crop."
    )


def _pi_compute_error(ground_truth: float, prediction_result: Dict) -> float:
    pred = prediction_result.get("personal_info_privacy_score", {}).get("score", 0.0)
    return abs(float(ground_truth) - float(pred))


def _pi_extract_summary(result: Dict) -> Dict:
    return result.get("personal_info_privacy_score", {})


def _pi_normalize_gt(raw) -> float:
    return float(raw)


def _pi_serialize_gt(value) -> float:
    return float(value)


def _pi_mask_level_label(mask_level: int) -> str:
    return f"severity<={int(mask_level)}"


# ============================================================================
# Social-relationship behaviors
# ============================================================================

#: Social-relationship features carry the same ``properties`` list that geolocation features do,
#: so the mask-level test is literally the same predicate. Aliased rather than copied, because two
#: copies of a filter is how a setting quietly starts protecting a different set of cues.
_sr_passes = _geo_passes


def _sr_pair_key(raw: Any) -> str | None:
    """
    Normalise whatever the model wrote for a pair into PISC's ``"i j"`` form.

    The schema asks for the key verbatim, but models reach for ``"P1-P2"``, ``"1-2"`` and
    ``"2 1"``. All three name the same annotated pair, and scoring them as unanswered would credit
    the defense for the model's formatting. Anything without exactly two numbers is rejected.
    """
    import re
    digits = re.findall(r"\d+", str(raw))
    if len(digits) != 2:
        return None
    first, second = int(digits[0]), int(digits[1])
    if first > second:
        first, second = second, first
    return f"{first} {second}"


def _sr_predicted_labels(result: Dict) -> Dict[str, Any]:
    """The ``{"i j": label}`` map a response asserts, keyed the way the ground truth is."""
    estimate = result.get("relationship_estimate") or {}
    predictions: Dict[str, Any] = {}
    for entry in estimate.get("pairs") or []:
        if not isinstance(entry, dict):
            continue
        key = _sr_pair_key(entry.get("pair"))
        if key is not None:
            predictions[key] = entry.get("label")
    return predictions


def _sr_format_r1_context(result: Dict) -> str:
    labels = _sr_predicted_labels(result)
    if not labels:
        return "no relationship reported for any pair"
    parts = []
    for key in sorted(labels, key=lambda k: [int(n) for n in k.split()]):
        first, second = key.split()
        parts.append(f"P{first}-P{second}: {labels[key]}")
    return "; ".join(parts)


def _sr_build_r2_prompt(r1_context: str, region_name: str, base_prompt: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"WHAT ROUND 1 CONCLUDED FROM THE UNEDITED IMAGE: {r1_context}\n"
        f"You are now re-reading region '{region_name}' of the edited image. Judge it on what is "
        f"visible now; the round-1 labels are context for what was removed, not an answer to "
        f"repeat."
    )


def _sr_compute_error(ground_truth: Dict[str, int], prediction_result: Dict) -> float:
    """
    Share of the image's annotated pairs the response fails to recover.

    0.0 means every annotated pair was classified correctly; 1.0 means none was. A pair the model
    left unanswered counts as not recovered -- see
    :meth:`veillens.metrics.relationship.PairScore.error_rate` for why silence must not read as
    protection, and where the other denominator is kept.
    """
    return pair_accuracy(ground_truth, _sr_predicted_labels(prediction_result)).error_rate


def _sr_extract_summary(result: Dict) -> Dict:
    """
    What lands in ``evaluation_results.json`` as this response's reading.

    Carries the raw estimate *and* the normalised ``predictions`` map, so ``veillens-score`` can
    recompute six-way and three-way accuracy from the results file alone, without re-parsing pair
    keys or reaching back into the attacker's cached responses.
    """
    estimate = result.get("relationship_estimate") or {}
    return {
        "pairs": estimate.get("pairs") or [],
        "predictions": _sr_predicted_labels(result),
    }


def _sr_normalize_gt(raw: Dict) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for key, value in (raw or {}).items():
        pair = _sr_pair_key(key)
        if pair is not None:
            normalized[pair] = int(value)
    return normalized


def _sr_serialize_gt(value: Dict[str, int]) -> Dict[str, int]:
    return {key: int(code) for key, code in value.items()}


def _sr_mask_level_label(mask_levels: List[str]) -> str:
    return ", ".join(mask_levels)


def _sr_format_item_context(item: Any) -> str | None:
    """
    The legend and pair list appended to every prompt for one image.

    Without this the prompt cannot say which people it is asking about: the boxes are drawn on the
    image, but only the manifest knows which numbered subjects form an annotated pair.
    """
    if not isinstance(item, dict):
        return None
    subjects = item.get("subjects") or {}
    pairs = item.get("ground_truth_relations") or {}
    if not subjects or not pairs:
        return None

    ids = sorted(subjects, key=lambda sid: int(sid))
    lines = [
        "",
        f"Subjects in this image are outlined and labelled "
        f"{', '.join('P' + sid for sid in ids)}.",
        "",
        "Pairs to analyse:",
    ]
    for key in sorted(pairs, key=lambda k: [int(n) for n in k.split()]):
        first, second = key.split()
        lines.append(f'- P{first} and P{second}   (pair key "{key}")')
    return "\n".join(lines)


def _sr_feature_boxes(feature: Dict, item: Any, options: Dict) -> Any:
    """
    The SAM 2 box prompt for one nominated subject: the dataset's person box, unmodified.

    Always the whole person, even under ``mask_scope: face``. SAM 2 is good at turning a person box
    into a person silhouette and has no idea what a head band cut out of that box is supposed to
    contain, so the narrowing happens afterwards, on the mask -- see `_sr_refine_mask`.

    Returns None when the feature names no subject this manifest row knows about, which sends the
    clue back down the Grounding DINO path rather than silently masking nothing.
    """
    if not isinstance(item, dict):
        return None
    subjects = item.get("subjects") or {}
    subject_id = feature.get("subject_id")
    if subject_id is None:
        return None
    box = subjects.get(str(subject_id))
    if box is None:
        return None

    # Imported here, not at module scope: this module is on the import path of `veillens.config`,
    # which has to load on a machine with no GPU stack.
    import numpy as np
    return np.array([[float(v) for v in box]], dtype=np.float32)


def _sr_refine_mask(mask: Any, feature: Dict, item: Any, options: Dict) -> Any:
    """
    Cut a person mask down to the head, which is what this setting's policy actually edits.

    ``mask_scope: person`` skips it and inpaints the whole silhouette -- stronger suppression, but
    not the paper's policy, and it destroys the posture and clothing that make the image worth
    keeping at all.
    """
    import numpy as np

    options = options or {}
    binary = np.asarray(mask) > 0

    # Clip to the subject's own box first. The policy is to anonymize "the face regions *within
    # the corresponding boxes*", and the dataset is authoritative about where this subject is --
    # so anything SAM 2 returned outside that box is bleed into the background or into the person
    # next to them, and keeping it would both over-edit the scene and mask the wrong subject.
    box = ((item or {}).get("subjects") or {}).get(str(feature.get("subject_id")))
    if box is not None:
        height, width = binary.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        keep = np.zeros_like(binary)
        top, bottom = max(int(min(y1, y2)), 0), min(int(round(max(y1, y2))) + 1, height)
        left, right = max(int(min(x1, x2)), 0), min(int(round(max(x1, x2))) + 1, width)
        keep[top:bottom, left:right] = True
        binary = binary & keep

    if str(options.get("mask_scope", "face")).strip().lower() == "person":
        return binary
    return head_band(
        binary,
        head_fraction=float(options.get("head_fraction", 0.30)),
        width_fraction=float(options.get("width_fraction", 1.0)),
        head_width_multiple=float(options.get("head_width_multiple", 1.6)),
    )


# ============================================================================
# Spec instances
# ============================================================================


GEOLOCATION_SPEC = PrivacyClassSpec(
    name="geolocation",
    anonymized_field="non_geolocation_element",
    privacy_field="geolocation_element",
    ground_truth_field="ground_truth_coords",
    gt_output_key="ground_truth_coords",
    error_key="error_km",
    error_increase_key="error_increase_km",
    error_unit="km",
    round1_schema_path="prompts/geolocation/schema_original.json",
    round1_prompt_path="prompts/geolocation/defense_base_prompt.txt",
    round2_schema_path="prompts/geolocation/schema_original.json",
    round2_prompt_path="prompts/geolocation/phase2_base_prompt.txt",
    eval_schema_path="prompts/geolocation/schema_eval.json",
    eval_prompt_path="prompts/geolocation/attack_base_prompt.txt",
    eval_schema_type="eval",
    feature_passes_mask_level=_geo_passes,
    format_round1_context=_geo_format_r1_context,
    build_round2_prompt=_geo_build_r2_prompt,
    compute_error=_geo_compute_error,
    extract_prediction_summary=_geo_extract_summary,
    normalize_ground_truth=_geo_normalize_gt,
    serialize_ground_truth=_geo_serialize_gt,
    format_mask_level_label=_geo_mask_level_label,
)


PERSONAL_INFO_SPEC = PrivacyClassSpec(
    name="compositional",
    anonymized_field="Anonymized_element",
    privacy_field="privacy_element",
    ground_truth_field="ground_truth_score",
    gt_output_key="ground_truth_score",
    error_key="error_score",
    error_increase_key="error_score_increase",
    error_unit="score",
    round1_schema_path="prompts/compositional/schema_original.json",
    round1_prompt_path="prompts/compositional/defense_base_prompt.txt",
    round2_schema_path="prompts/compositional/schema_original.json",
    round2_prompt_path="prompts/compositional/defense_base_prompt.txt",
    eval_schema_path="prompts/compositional/schema_original.json",
    eval_prompt_path="prompts/compositional/defense_base_prompt.txt",
    feature_passes_mask_level=_pi_passes,
    format_round1_context=_pi_format_r1_context,
    build_round2_prompt=_pi_build_r2_prompt,
    compute_error=_pi_compute_error,
    extract_prediction_summary=_pi_extract_summary,
    normalize_ground_truth=_pi_normalize_gt,
    serialize_ground_truth=_pi_serialize_gt,
    format_mask_level_label=_pi_mask_level_label,
    # VISPR ships unresized camera originals -- up to ~10 megapixels. Inpainting one at native
    # resolution OOMs a 48 GB A40. Geolocation (Cityscapes, 2048x1024) and social relationship
    # (PISC, web-sized) are already under this, so only this setting needs the cap.
    max_input_dim=1024,
)


SOCIAL_RELATION_SPEC = PrivacyClassSpec(
    name="social_relation",
    anonymized_field="anonymized_element",
    privacy_field="relationship_element",
    ground_truth_field="ground_truth_relations",
    gt_output_key="ground_truth_relations",
    error_key="error_relation",
    error_increase_key="error_relation_increase",
    error_unit="pair",
    round1_schema_path="prompts/social_relation/schema_original.json",
    round1_prompt_path="prompts/social_relation/defense_base_prompt.txt",
    round2_schema_path="prompts/social_relation/schema_original.json",
    round2_prompt_path="prompts/social_relation/phase2_base_prompt.txt",
    eval_schema_path="prompts/social_relation/schema_eval.json",
    eval_prompt_path="prompts/social_relation/attack_base_prompt.txt",
    feature_passes_mask_level=_sr_passes,
    format_round1_context=_sr_format_r1_context,
    build_round2_prompt=_sr_build_r2_prompt,
    compute_error=_sr_compute_error,
    extract_prediction_summary=_sr_extract_summary,
    normalize_ground_truth=_sr_normalize_gt,
    serialize_ground_truth=_sr_serialize_gt,
    format_mask_level_label=_sr_mask_level_label,
    eval_schema_type="eval",
    # The headline number is one adversary's accuracy before versus after, so the "before" reading
    # has to come from that adversary and not from the defense oracle's round-1 response.
    eval_original=True,
    feature_boxes=_sr_feature_boxes,
    refine_mask=_sr_refine_mask,
    format_item_context=_sr_format_item_context,
)


def get_spec(config: Dict[str, Any]) -> PrivacyClassSpec:
    """
    Resolve the PrivacyClassSpec for the given pipeline config.

    Default paths come from the registered specs; they can be overridden per run
    via a top-level ``prompts:`` section in the config YAML.  All values are
    paths (with extension) relative to the repository root::

        prompts:
          round1_schema: "prompts/geolocation/schema_original.json"
          round1_prompt: "prompts/geolocation/defense_base_prompt.txt"
          round2_schema: "prompts/geolocation/schema_original.json"
          round2_prompt: "prompts/geolocation/phase2_base_prompt.txt"
          eval_schema:   "prompts/geolocation/schema_eval.json"
          eval_prompt:   "prompts/geolocation/attack_base_prompt.txt"

    Any unspecified key keeps the spec's default.
    """
    from veillens.privacy.registry import _SPECS, canonical_name

    name = canonical_name(
        config.get("experiment", {}).get("privacy_class", "geolocation")
    )
    if name not in _SPECS:
        raise ValueError(
            f"Unknown experiment.privacy_class: {name!r}. "
            f"Expected one of: {sorted(_SPECS)}"
        )
    base = _SPECS[name]

    overrides = config.get("prompts") or {}
    if not overrides:
        return base

    from dataclasses import replace
    return replace(
        base,
        round1_schema_path=overrides.get("round1_schema", base.round1_schema_path),
        round1_prompt_path=overrides.get("round1_prompt", base.round1_prompt_path),
        round2_schema_path=overrides.get("round2_schema", base.round2_schema_path),
        round2_prompt_path=overrides.get("round2_prompt", base.round2_prompt_path),
        eval_schema_path=overrides.get("eval_schema", base.eval_schema_path),
        eval_prompt_path=overrides.get("eval_prompt", base.eval_prompt_path),
        _schemas=None,
        _prompts=None,
    )


#: Paper vocabulary. `PERSONAL_INFO_SPEC` remains as the historical symbol name.
COMPOSITIONAL_SPEC = PERSONAL_INFO_SPEC
