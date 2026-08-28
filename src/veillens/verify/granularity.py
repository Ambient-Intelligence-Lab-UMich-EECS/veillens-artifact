"""
The granularity ladder, and what it means for an image to be protected at a given rung.

A run is *configured* by granularity: ``experiment.mask_level`` says which cue granularities the
loop removes. This module is the other half of that contract -- it derives, from the same
``mask_level``, what must be true of the finished image for the run to have delivered what it
promised. Nothing here reads a config file, calls a model, or touches the GPU; it is pure
functions over the run's own JSON so that verification can be recomputed offline, years later,
from an archived run.

Two independent criteria, because they can disagree and the disagreement is informative:

* **metric** -- the attacker's point estimate must be at least ``threshold_km`` from the ground
  truth. This is the paper's criterion ("city-level protection, treating predictions within 10 km
  as privacy failures"). It needs the dataset label, so it is an offline measurement, never a
  control signal a deployed system could use.
* **residual** -- the attacker's own account of what it used must contain no cue tagged at a
  protected granularity. This needs no ground truth at all: it reads the granularity labels the
  attacker already emits under ``schema_eval.json``. It is the stricter of the two, and it is the
  one that is actually *about* granularity rather than merely parameterised by it.

Both are three-valued. A criterion that could not be computed is ``None``, and ``None`` never
becomes a pass -- missing evidence is not evidence of protection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "LADDER_KM",
    "CLUE_LEVEL_TO_RUNG",
    "LEVELS_BY_SPECIFICITY",
    "ResidualCue",
    "ImageVerdict",
    "coarsest_level",
    "target_rung",
    "threshold_km",
    "protected_levels",
    "residual_cues",
    "verdict",
    "three_valued_and",
]

GEOLOCATION = "geolocation"
COMPOSITIONAL = "compositional"

#: The reporting ladder, from the paper's threat model. Attack success is reported at each rung;
#: the rung a run *targets* is derived from its ``mask_level`` (see :func:`target_rung`).
LADDER_KM: Dict[str, float] = {
    "block": 0.5,
    "street": 1.0,
    "neighborhood": 5.0,
    "city": 10.0,
    "regional": 100.0,
}

#: Which rung each clue granularity protects to. The ladder has five rungs but the oracle schema
#: has three clue levels, so ``block`` and ``neighborhood`` are reporting-only: they appear in the
#: threshold table and can never be a target. See docs/VERIFICATION.md.
CLUE_LEVEL_TO_RUNG: Dict[str, str] = {
    "street level clue": "street",
    "city level clue": "city",
    "country level clue": "regional",
}

#: Most specific first. Deliberately duplicated from ``veillens.loop.state`` rather than imported:
#: ``loop.api_workers`` imports this module, so importing back would be a cycle, and ``loop.state``
#: pulls in numpy which this module has no other need for. ``test_verify_granularity`` asserts the
#: two definitions stay identical.
LEVELS_BY_SPECIFICITY: Tuple[str, ...] = (
    "street level clue",
    "city level clue",
    "country level clue",
)

#: CPRT severity runs 1 (most severe) to 4 (least). ``mask_level: 2`` protects levels 1 and 2.
COMPOSITIONAL_SEVERITIES: Tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class ResidualCue:
    """One cue the attacker still reported, at a granularity the run promised to remove."""

    description: str
    #: The level(s) it was tagged with: clue-level strings for geolocation, a severity int for
    #: compositional, or ``None`` when the label was missing or unparseable.
    level: Union[Tuple[str, ...], int, None]
    #: True when this cue counts as residual only because its granularity could not be read.
    malformed: bool = False

    def as_json(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "level": list(self.level) if isinstance(self.level, tuple) else self.level,
            "malformed": self.malformed,
        }


@dataclass(frozen=True)
class ImageVerdict:
    """Both criteria for one image, plus the conjunction."""

    image_id: str
    target_rung: Optional[str]
    threshold_km: Optional[float]
    error: Optional[float]
    metric_pass: Optional[bool]
    residual: Tuple[ResidualCue, ...] = ()
    residual_pass: Optional[bool] = None
    passed: Optional[bool] = None
    reason: str = ""

    def as_json(self) -> Dict[str, Any]:
        return {
            "target_rung": self.target_rung,
            "threshold_km": self.threshold_km,
            "error": self.error,
            "metric_pass": self.metric_pass,
            "residual_pass": self.residual_pass,
            "residual_cues": [cue.as_json() for cue in self.residual],
            "passed": self.passed,
            "reason": self.reason,
        }


def three_valued_and(*values: Optional[bool]) -> Optional[bool]:
    """
    Kleene AND: any ``False`` wins, then any ``None``, else ``True``.

    The ordering is the whole point. A criterion that could not be computed must not be able to
    turn a failure into a pass, and must not silently become a pass on its own.
    """
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


def coarsest_level(mask_level: Sequence[str]) -> Optional[str]:
    """
    The least specific clue level in *mask_level* -- the granularity the run protects *to*.

    ``["street level clue", "city level clue"]`` -> ``"city level clue"``: removing street and
    city cues is city-level protection, so the run must be verified at the city rung.
    """
    present = {str(level).strip().lower() for level in mask_level or ()}
    for level in reversed(LEVELS_BY_SPECIFICITY):
        if level in present:
            return level
    return None


def target_rung(privacy_class: str, mask_level: Any) -> Optional[str]:
    """The ladder rung this run targets, or ``None`` when its metric is not a distance."""
    if privacy_class != GEOLOCATION:
        # Compositional risk is a unitless CPRT score, not a distance; it has no rung. Its
        # verification is the residual criterion alone (see docs/VERIFICATION.md).
        return None
    coarsest = coarsest_level(mask_level if isinstance(mask_level, (list, tuple)) else [])
    return CLUE_LEVEL_TO_RUNG.get(coarsest) if coarsest else None


def threshold_km(privacy_class: str, mask_level: Any) -> Optional[float]:
    """The distance an attacker must be wrong by for this run to count as protected."""
    rung = target_rung(privacy_class, mask_level)
    return LADDER_KM[rung] if rung else None


def protected_levels(privacy_class: str, mask_level: Any) -> frozenset:
    """The granularities the run promised to remove, as a set to test cues against."""
    if privacy_class == GEOLOCATION:
        if not isinstance(mask_level, (list, tuple)):
            return frozenset()
        return frozenset(
            str(level).strip().lower()
            for level in mask_level
            if str(level).strip().lower() in CLUE_LEVEL_TO_RUNG
        )
    try:
        ceiling = int(mask_level)
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(level for level in COMPOSITIONAL_SEVERITIES if level <= ceiling)


def _features(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    critical = result.get("critical_features") or {}
    if not isinstance(critical, Mapping):
        return []
    features = critical.get("features")
    return [f for f in features if isinstance(f, Mapping)] if isinstance(features, list) else []


def _describe(feature: Mapping[str, Any]) -> str:
    for key in ("geolocation_description", "object_description", "privacy_element",
                "geolocation_element", "target_concept"):
        value = feature.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "<undescribed cue>"


def _geo_levels(feature: Mapping[str, Any]) -> Optional[Tuple[str, ...]]:
    raw = feature.get("properties")
    if not isinstance(raw, list) or not raw:
        return None
    levels = tuple(
        str(item).strip().lower() for item in raw
        if isinstance(item, str) and str(item).strip().lower() in CLUE_LEVEL_TO_RUNG
    )
    return levels or None


def _pi_severity(feature: Mapping[str, Any]) -> Optional[int]:
    raw = feature.get("severity_level")
    if isinstance(raw, bool):
        return None
    try:
        severity = int(raw)
    except (TypeError, ValueError):
        return None
    return severity if severity in COMPOSITIONAL_SEVERITIES else None


def residual_cues(result: Optional[Mapping[str, Any]], privacy_class: str,
                  mask_level: Any) -> Optional[List[ResidualCue]]:
    """
    The cues the attacker still reported at a protected granularity.

    Returns ``None`` when *result* is absent -- "not computed", which is different from "none
    found" and must not be reported as a pass.

    Fails closed: a cue whose granularity label is missing or unparseable is counted as residual
    and flagged ``malformed``. A well-formed cue at an out-of-scope granularity (a country-level
    cue under city-level protection) is not residual -- the run never promised to remove it.
    """
    if result is None:
        return None
    if not isinstance(result, Mapping):
        return None

    protected = protected_levels(privacy_class, mask_level)
    found: List[ResidualCue] = []

    for feature in _features(result):
        description = _describe(feature)
        if privacy_class == GEOLOCATION:
            levels = _geo_levels(feature)
            if levels is None:
                found.append(ResidualCue(description, None, malformed=True))
            elif protected.intersection(levels):
                found.append(ResidualCue(description, levels))
        else:
            severity = _pi_severity(feature)
            if severity is None:
                found.append(ResidualCue(description, None, malformed=True))
            elif severity in protected:
                found.append(ResidualCue(description, severity))

    return found


def verdict(image_id: str, *, privacy_class: str, mask_level: Any,
            error: Optional[float], attacker_result: Optional[Mapping[str, Any]]) -> ImageVerdict:
    """Both criteria for one image. See the module docstring for what each means."""
    rung = target_rung(privacy_class, mask_level)
    limit = threshold_km(privacy_class, mask_level)

    metric_pass: Optional[bool] = None
    if limit is not None and error is not None:
        metric_pass = float(error) >= limit

    cues = residual_cues(attacker_result, privacy_class, mask_level)
    residual_pass: Optional[bool] = None if cues is None else not cues

    # "Not applicable" is not "not computed". A privacy class with no rung on the distance ladder
    # has no metric criterion to fail, so the residual criterion is the whole verdict; folding an
    # inapplicable criterion in as unknown would report every compositional image as unverifiable.
    # A class that *does* have a rung but produced no error is genuinely unknown, and propagates.
    if limit is None:
        passed = residual_pass
    else:
        passed = three_valued_and(metric_pass, residual_pass)

    return ImageVerdict(
        image_id=image_id,
        target_rung=rung,
        threshold_km=limit,
        error=float(error) if error is not None else None,
        metric_pass=metric_pass,
        residual=tuple(cues or ()),
        residual_pass=residual_pass,
        passed=passed,
        reason=_reason(metric_pass, residual_pass, error, limit, cues, privacy_class),
    )


def _reason(metric_pass: Optional[bool], residual_pass: Optional[bool], error: Optional[float],
            limit: Optional[float], cues: Optional[List[ResidualCue]],
            privacy_class: str) -> str:
    parts: List[str] = []
    if metric_pass is False:
        parts.append(f"attacker within {limit:g} km ({error:.2f} km)")
    elif metric_pass is None:
        if limit is None:
            parts.append("no distance rung for this privacy class")
        else:
            parts.append("no attacker error recorded")

    if residual_pass is False and cues:
        malformed = sum(1 for cue in cues if cue.malformed)
        clean = len(cues) - malformed
        if clean:
            parts.append(f"{clean} residual cue(s) at a protected granularity")
        if malformed:
            parts.append(f"{malformed} cue(s) with an unreadable granularity label")
    elif residual_pass is None:
        parts.append("no attacker response found, residual check not computed")

    return "; ".join(parts) if parts else "passed both criteria"
