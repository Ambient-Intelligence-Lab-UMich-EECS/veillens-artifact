"""
Privacy class specifications.

A `PrivacyClassSpec` bundles everything that differs between privacy classes — prompt and schema
paths, which cues count as in-scope for a given mask level, how round-2 context is phrased, and
what "error" means — so the loop can stay generic instead of branching on a class name.

This is the extension point of the whole system. To add a fourth privacy class, see
`docs/EXTENDING.md`; you write one spec instance and register it, and nothing in
`veillens.loop` changes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from veillens.metrics.geodesic import DistanceCalculator
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
