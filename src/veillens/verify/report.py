"""
Aggregate per-image verdicts into a run-level verification report.

Everything here works from a run's own JSON, so a finished run can be re-verified offline with no
API key, no GPU and no model weights -- including runs archived before this module existed.

Two inputs, and they come from different files:

* ``evaluation_results.json`` carries the attacker's error per image, which is the metric
  criterion. It does *not* carry the attacker's cue list -- ``gpt_response`` is only the
  ``location_estimate`` sub-object -- so it cannot answer the granularity question on its own.
* the raw attacker responses (``eval_cache/*_eval.json`` in a live run,
  ``evaluation/<model>/responses/*.json`` in an archived one) carry
  ``critical_features[].properties``, which is the residual criterion.

When the second is missing the report says so per image and in the summary. It never treats a
missing response as a clean one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from veillens.verify.granularity import (
    CLUE_LEVEL_TO_RUNG,
    COMPOSITIONAL,
    GEOLOCATION,
    LADDER_KM,
    ImageVerdict,
    protected_levels,
    target_rung,
    threshold_km,
    verdict,
)

__all__ = [
    "RunVerification",
    "verify_run",
    "infer_mask_level",
    "sniff_privacy_class",
    "load_responses",
    "find_response_dirs",
]

_SEVERITY_LABEL = re.compile(r"severity\s*<=\s*(\d+)")


# ---------------------------------------------------------------------------
# Reading a run's own record of what it did
# ---------------------------------------------------------------------------


def sniff_privacy_class(evaluation_results: Mapping[str, Any]) -> str:
    """Geolocation runs key their error as ``error_km``; compositional as ``error_score``."""
    for entry in evaluation_results.values():
        if not isinstance(entry, Mapping):
            continue
        for item in entry.get("feature_results") or []:
            if "error_km" in item:
                return GEOLOCATION
            if "error_score" in item:
                return COMPOSITIONAL
    return GEOLOCATION


def _protected_entry(entry: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The final protected reading: the last ``feature_results`` item that is not the original."""
    for item in reversed(entry.get("feature_results") or []):
        if isinstance(item, Mapping) and item.get("feature_removed") != "none":
            return item
    return None


def infer_mask_level(evaluation_results: Mapping[str, Any], privacy_class: str) -> Any:
    """
    Recover ``experiment.mask_level`` from the run's own results.

    The loop writes ``feature_removed`` as ``PrivacyClassSpec.format_mask_level_label(mask_level)``
    -- ``"street level clue, city level clue"`` for geolocation, ``"severity<=2"`` for
    compositional. So a run records the granularity it targeted, and an archived run can be
    verified against its own promise without needing the config that produced it.
    """
    for entry in evaluation_results.values():
        if not isinstance(entry, Mapping):
            continue
        item = _protected_entry(entry)
        if item is None:
            continue
        label = item.get("feature_removed")
        if not isinstance(label, str):
            continue
        if privacy_class == GEOLOCATION:
            levels = [part.strip().lower() for part in label.split(",")]
            levels = [level for level in levels if level in CLUE_LEVEL_TO_RUNG]
            if levels:
                return levels
        else:
            match = _SEVERITY_LABEL.search(label)
            if match:
                return int(match.group(1))
    return [] if privacy_class == GEOLOCATION else None


def _error_of(entry: Mapping[str, Any], privacy_class: str) -> Optional[float]:
    item = _protected_entry(entry)
    if item is None:
        return None
    key = "error_km" if privacy_class == GEOLOCATION else "error_score"
    value = item.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Locating the attacker's raw responses
# ---------------------------------------------------------------------------


def find_response_dirs(run_dir: Path) -> List[Path]:
    """
    Directories under *run_dir* that hold one attacker response per image.

    Covers the live layout (``eval_cache/``) and the archived one
    (``evaluation/<model>/responses/``) without hardcoding either depth.
    """
    found: List[Path] = []
    for pattern in ("eval_cache", "**/eval_cache", "**/responses"):
        for candidate in run_dir.glob(pattern):
            if candidate.is_dir() and any(candidate.glob("*.json")) and candidate not in found:
                found.append(candidate)
    return sorted(found)


def load_responses(directory: Path) -> Dict[str, Dict[str, Any]]:
    """Index one response directory by image stem, dropping the ``_eval`` suffix if present."""
    index: Dict[str, Dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.stem[:-5] if path.stem.endswith("_eval") else path.stem
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            index[stem] = dict(payload)
    return index


def _match(image_id: str, index: Mapping[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find this image's response.

    Exact first. Failing that, allow one side to be a prefix of the other, because the loop names
    caches after the image *file* stem while results are keyed by image id -- Cityscapes ids gain
    a ``_leftImg8bit`` suffix on the way to disk. Ties break toward the closest match.
    """
    if image_id in index:
        return index[image_id]
    candidates = [
        key for key in index
        if key.startswith(image_id) or image_id.startswith(key)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda key: abs(len(key) - len(image_id)))
    return index[candidates[0]]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class RunVerification:
    privacy_class: str
    mask_level: Any
    target_rung: Optional[str]
    threshold_km: Optional[float]
    residual_source: Optional[str]
    verdicts: Dict[str, ImageVerdict] = field(default_factory=dict)
    matched: int = 0

    # -- tallies ------------------------------------------------------------

    def _count(self, attribute: str, value: Optional[bool]) -> int:
        return sum(1 for v in self.verdicts.values() if getattr(v, attribute) is value)

    @property
    def n(self) -> int:
        return len(self.verdicts)

    @property
    def disagreements(self) -> Tuple[int, int]:
        """(metric passes but residual fails, residual passes but metric fails)."""
        metric_only = sum(
            1 for v in self.verdicts.values()
            if v.metric_pass is True and v.residual_pass is False
        )
        residual_only = sum(
            1 for v in self.verdicts.values()
            if v.residual_pass is True and v.metric_pass is False
        )
        return metric_only, residual_only

    def residual_by_level(self) -> Dict[str, int]:
        """
        Residual cues tallied by the granularity that makes them residual.

        A cue tagged both city- and country-level under city protection is counted once, as
        city. Counting its country tag as well would make an out-of-scope granularity look like
        a leak.
        """
        protected = protected_levels(self.privacy_class, self.mask_level)
        tally: Dict[str, int] = {}
        for image in self.verdicts.values():
            for cue in image.residual:
                if cue.malformed:
                    tally["unreadable label"] = tally.get("unreadable label", 0) + 1
                elif isinstance(cue.level, tuple):
                    for level in cue.level:
                        if level in protected:
                            tally[level] = tally.get(level, 0) + 1
                else:
                    key = f"severity {cue.level}"
                    tally[key] = tally.get(key, 0) + 1
        return tally

    def ladder(self) -> Dict[str, float]:
        """Attack success at each rung: the share of protected images localised within it."""
        errors = [v.error for v in self.verdicts.values() if v.error is not None]
        if not errors or self.privacy_class != GEOLOCATION:
            return {}
        return {
            rung: 100.0 * sum(1 for e in errors if e < km) / len(errors)
            for rung, km in LADDER_KM.items()
        }

    # -- serialisation ------------------------------------------------------

    def as_json(self) -> Dict[str, Any]:
        metric_only, residual_only = self.disagreements
        return {
            "privacy_class": self.privacy_class,
            "mask_level": self.mask_level,
            "target_rung": self.target_rung,
            "threshold_km": self.threshold_km,
            "ladder_km": LADDER_KM,
            "protected_levels": sorted(
                protected_levels(self.privacy_class, self.mask_level), key=str
            ),
            "criterion": {
                "metric": (
                    f"attacker error >= {self.threshold_km} km, derived from mask_level"
                    if self.threshold_km is not None
                    else "not applicable: this privacy class has no distance rung"
                ),
                "residual": "no cue at a protected granularity survives in the attacker's report",
                "combined": "both must hold; a criterion that could not be computed is not a pass",
            },
            "residual_source": self.residual_source,
            "images_matched_to_a_response": self.matched,
            "n": self.n,
            "summary": {
                "passed": self._count("passed", True),
                "failed": self._count("passed", False),
                "not_computed": self._count("passed", None),
                "metric_pass": self._count("metric_pass", True),
                "metric_fail": self._count("metric_pass", False),
                "metric_not_computed": self._count("metric_pass", None),
                "residual_pass": self._count("residual_pass", True),
                "residual_fail": self._count("residual_pass", False),
                "residual_not_computed": self._count("residual_pass", None),
                "metric_passes_residual_fails": metric_only,
                "residual_passes_metric_fails": residual_only,
            },
            "residual_by_level": self.residual_by_level(),
            "attack_success_pct_by_rung": {
                rung: round(pct, 1) for rung, pct in self.ladder().items()
            },
            "images": {
                image_id: image.as_json() for image_id, image in sorted(self.verdicts.items())
            },
        }

    def write(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_json(), indent=2) + "\n")
        return destination

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        lines: List[str] = []
        out = lines.append

        label = "Geolocation" if self.privacy_class == GEOLOCATION else "Compositional privacy"
        out(f"\nVerification — {label}, {self.n} images\n")

        if self.threshold_km is not None:
            out(f"  target granularity : {self.target_rung} "
                f"(attacker must be ≥ {self.threshold_km:g} km off)")
        else:
            out("  target granularity : n/a — no distance rung for this privacy class; "
                "the residual criterion is the verdict")
        protected = sorted(protected_levels(self.privacy_class, self.mask_level), key=str)
        out(f"  protected cues     : {', '.join(str(p) for p in protected) or '(none)'}")
        if self.residual_source:
            out(f"  residual source    : {self.residual_source} "
                f"({self.matched}/{self.n} images matched)")
        elif self.matched:
            out(f"  residual source    : supplied directly "
                f"({self.matched}/{self.n} images matched)")
        else:
            out("  residual source    : none found — residual criterion NOT computed")

        out("")
        out(f"  {'':38s}{'pass':>8s}{'fail':>8s}{'not computed':>15s}")
        out("  " + "-" * 67)
        metric_label = (
            f"metric   (error ≥ {self.threshold_km:g} km)" if self.threshold_km is not None
            else "metric   (not applicable)"
        )
        for name, attribute in ((metric_label, "metric_pass"),
                                ("residual (no protected cue survives)", "residual_pass"),
                                ("both", "passed")):
            if name == "both":
                out("  " + "-" * 67)
            out(f"  {name:38s}{self._count(attribute, True):>8d}"
                f"{self._count(attribute, False):>8d}{self._count(attribute, None):>15d}")

        metric_only, residual_only = self.disagreements
        if metric_only or residual_only:
            out("")
            out(f"  criteria disagree on {metric_only + residual_only} images "
                f"(metric passes / residual fails: {metric_only}; converse: {residual_only})")

        if self.n > 1 and self._count("residual_pass", False) == self.n:
            out("")
            out("  ! every image failed the residual criterion. Before reading that as leakage,")
            out("    check that the attacker prompt permits an empty critical_features list: a")
            out("    prompt that mandates a clue at a protected granularity makes this criterion")
            out("    unpassable by construction. See docs/VERIFICATION.md.")

        residual = self.residual_by_level()
        if residual:
            out("")
            out("  residual cues at a protected granularity")
            for level, count in sorted(residual.items(), key=lambda kv: -kv[1]):
                out(f"    {level:24s}{count:>6d}")

        ladder = self.ladder()
        if ladder:
            out("")
            out("  attack success on the protected images, at each rung")
            for rung, km in LADDER_KM.items():
                marker = "  ← target" if rung == self.target_rung else ""
                out(f"    {rung:>12s} {km:>6g} km   {ladder[rung]:5.1f}%{marker}")

        out("")
        return "\n".join(lines)


def verify_run(evaluation_results: Mapping[str, Any],
               attacker_responses: Optional[Mapping[str, Mapping[str, Any]]] = None,
               *,
               privacy_class: Optional[str] = None,
               mask_level: Any = None,
               residual_source: Optional[str] = None) -> RunVerification:
    """
    Verify every image in *evaluation_results* against the granularity the run targeted.

    *privacy_class* and *mask_level* are recovered from the results themselves when not given,
    so this works on an archived run with no config beside it.
    """
    privacy_class = privacy_class or sniff_privacy_class(evaluation_results)
    if mask_level is None:
        mask_level = infer_mask_level(evaluation_results, privacy_class)

    report = RunVerification(
        privacy_class=privacy_class,
        mask_level=mask_level,
        target_rung=target_rung(privacy_class, mask_level),
        threshold_km=threshold_km(privacy_class, mask_level),
        residual_source=residual_source,
    )

    for image_id, entry in evaluation_results.items():
        if not isinstance(entry, Mapping):
            continue
        response = _match(image_id, attacker_responses) if attacker_responses else None
        if response is not None:
            report.matched += 1
        report.verdicts[image_id] = verdict(
            image_id,
            privacy_class=privacy_class,
            mask_level=mask_level,
            error=_error_of(entry, privacy_class),
            attacker_result=response,
        )

    return report
