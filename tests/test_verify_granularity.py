"""
Stage IV: the granularity contract.

The bug this module exists to prevent is drift between what a run *removes* and what it is
*checked against*. So the tests that matter most are the ones pinning that derivation: given a
`mask_level`, exactly one rung and exactly one distance follow, and there is no second place to
set either.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from veillens.config import load_config
from veillens.loop.state import LEVELS_BY_SPECIFICITY as STATE_LEVELS
from veillens.verify.granularity import (
    CLUE_LEVEL_TO_RUNG,
    LADDER_KM,
    LEVELS_BY_SPECIFICITY,
    coarsest_level,
    protected_levels,
    residual_cues,
    target_rung,
    three_valued_and,
    threshold_km,
    verdict,
)
from veillens.verify.report import (
    infer_mask_level,
    sniff_privacy_class,
    verify_run,
)

GEO = "geolocation"
COMP = "compositional"
CITY_PROTECTION = ["street level clue", "city level clue"]


# ---------------------------------------------------------------------------
# The ladder is a claim about the paper, not an implementation detail
# ---------------------------------------------------------------------------


def test_ladder_matches_the_papers_thresholds():
    """block 0.5 / street 1 / neighborhood 5 / city 10 / regional 100 km."""
    assert LADDER_KM == {
        "block": 0.5,
        "street": 1.0,
        "neighborhood": 5.0,
        "city": 10.0,
        "regional": 100.0,
    }


def test_level_ordering_matches_the_loops_own_ordering():
    """
    `verify.granularity` redefines the specificity order rather than importing it, because
    `loop.api_workers` imports this module and the reverse import would be a cycle. That makes
    the duplication load-bearing, so it is pinned here.
    """
    assert LEVELS_BY_SPECIFICITY == STATE_LEVELS


def test_every_clue_level_maps_to_a_real_rung():
    assert set(CLUE_LEVEL_TO_RUNG) == set(LEVELS_BY_SPECIFICITY)
    assert set(CLUE_LEVEL_TO_RUNG.values()) <= set(LADDER_KM)


# ---------------------------------------------------------------------------
# mask_level -> rung -> km, the derivation the whole design rests on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_level,rung,km", [
    (["street level clue"], "street", 1.0),
    (["street level clue", "city level clue"], "city", 10.0),
    (["city level clue"], "city", 10.0),
    (["street level clue", "city level clue", "country level clue"], "regional", 100.0),
])
def test_target_is_the_coarsest_protected_level(mask_level, rung, km):
    assert target_rung(GEO, mask_level) == rung
    assert threshold_km(GEO, mask_level) == km


def test_shipped_geolocation_config_targets_city_at_10km():
    """The load-bearing case: the shipped config is city-level protection, so 10 km, not 5."""
    config = load_config("configs/geolocation.yaml", check_dataset=False)
    mask_level = config["experiment"]["mask_level"]
    assert target_rung(GEO, mask_level) == "city"
    assert threshold_km(GEO, mask_level) == 10.0


def test_order_and_case_do_not_change_the_target():
    assert coarsest_level(["City Level Clue", "street level clue"]) == "city level clue"
    assert coarsest_level(reversed(CITY_PROTECTION)) == "city level clue"


def test_unknown_or_empty_mask_level_yields_no_rung():
    assert target_rung(GEO, []) is None
    assert target_rung(GEO, ["neighbourhood level clue"]) is None
    assert threshold_km(GEO, []) is None


def test_compositional_has_no_distance_rung():
    assert target_rung(COMP, 2) is None
    assert threshold_km(COMP, 2) is None
    assert protected_levels(COMP, 2) == frozenset({1, 2})
    assert protected_levels(COMP, 4) == frozenset({1, 2, 3, 4})


# ---------------------------------------------------------------------------
# The residual criterion
# ---------------------------------------------------------------------------


def _geo_result(*levels_per_feature):
    return {
        "location_estimate": {"latitude": 0.0, "longitude": 0.0},
        "critical_features": {
            "features": [
                {"geolocation_description": f"cue {i}", "properties": list(levels)}
                for i, levels in enumerate(levels_per_feature)
            ]
        },
    }


def test_out_of_scope_cue_is_not_residual():
    """A country-level cue survives city-level protection legitimately: it was never targeted."""
    cues = residual_cues(_geo_result(["country level clue"]), GEO, CITY_PROTECTION)
    assert cues == []


def test_protected_cue_is_residual():
    cues = residual_cues(_geo_result(["city level clue"]), GEO, CITY_PROTECTION)
    assert len(cues) == 1
    assert cues[0].level == ("city level clue",)
    assert not cues[0].malformed


def test_cue_tagged_at_several_levels_counts_if_any_is_protected():
    cues = residual_cues(
        _geo_result(["city level clue", "country level clue"]), GEO, CITY_PROTECTION)
    assert len(cues) == 1


def test_missing_or_garbage_level_fails_closed():
    """Unreadable evidence must never buy a pass."""
    for feature in ({"geolocation_description": "x"},
                    {"geolocation_description": "x", "properties": []},
                    {"geolocation_description": "x", "properties": ["planet level clue"]}):
        result = {"critical_features": {"features": [feature]}}
        cues = residual_cues(result, GEO, CITY_PROTECTION)
        assert len(cues) == 1, feature
        assert cues[0].malformed is True


def test_absent_response_is_not_computed_not_a_pass():
    assert residual_cues(None, GEO, CITY_PROTECTION) is None
    v = verdict("img", privacy_class=GEO, mask_level=CITY_PROTECTION,
                error=999.0, attacker_result=None)
    assert v.metric_pass is True
    assert v.residual_pass is None
    assert v.passed is None, "a criterion that could not be computed must not pass the image"


def test_compositional_residual_uses_severity():
    result = {"critical_features": {"features": [
        {"object_description": "a", "severity_level": 1},
        {"object_description": "b", "severity_level": 3},
        {"object_description": "c", "severity_level": "nonsense"},
    ]}}
    cues = residual_cues(result, COMP, 2)
    assert [c.level for c in cues] == [1, None]
    assert cues[1].malformed is True


# ---------------------------------------------------------------------------
# Combining the two criteria
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("values,expected", [
    ((True, True), True),
    ((True, False), False),
    ((False, None), False),
    ((True, None), None),
    ((None, None), None),
])
def test_three_valued_and(values, expected):
    assert three_valued_and(*values) is expected


def test_verdict_combines_both_criteria():
    clean = _geo_result(["country level clue"])
    leaky = _geo_result(["street level clue"])

    passed = verdict("a", privacy_class=GEO, mask_level=CITY_PROTECTION,
                     error=42.0, attacker_result=clean)
    assert (passed.metric_pass, passed.residual_pass, passed.passed) == (True, True, True)

    near = verdict("b", privacy_class=GEO, mask_level=CITY_PROTECTION,
                   error=3.0, attacker_result=clean)
    assert (near.metric_pass, near.residual_pass, near.passed) == (False, True, False)
    assert "3.00 km" in near.reason

    residual_only = verdict("c", privacy_class=GEO, mask_level=CITY_PROTECTION,
                            error=42.0, attacker_result=leaky)
    assert (residual_only.metric_pass, residual_only.residual_pass) == (True, False)
    assert residual_only.passed is False


def test_threshold_boundary_is_inclusive():
    """`error >= threshold` passes: exactly 10 km away is protected at the city rung."""
    exact = verdict("x", privacy_class=GEO, mask_level=CITY_PROTECTION,
                    error=10.0, attacker_result=None)
    assert exact.metric_pass is True
    just_under = verdict("x", privacy_class=GEO, mask_level=CITY_PROTECTION,
                         error=9.999, attacker_result=None)
    assert just_under.metric_pass is False


# ---------------------------------------------------------------------------
# Reading a run back
# ---------------------------------------------------------------------------


def _evaluation_results(label, error_key, errors):
    return {
        f"img{i}": {
            "feature_results": [
                {"feature_removed": "none", error_key: 0.1},
                {"feature_removed": label, error_key: err},
            ]
        }
        for i, err in enumerate(errors)
    }


def test_mask_level_is_recoverable_from_the_run_itself():
    """
    The loop writes `feature_removed` as the formatted mask level, so an archived run carries
    the promise it should be checked against -- no config needed.
    """
    geo = _evaluation_results("street level clue, city level clue", "error_km", [1.0])
    assert sniff_privacy_class(geo) == GEO
    assert infer_mask_level(geo, GEO) == CITY_PROTECTION

    comp = _evaluation_results("severity<=2", "error_score", [0.1])
    assert sniff_privacy_class(comp) == COMP
    assert infer_mask_level(comp, COMP) == 2


def test_verify_run_tallies_and_reports_disagreement():
    data = _evaluation_results(
        "street level clue, city level clue", "error_km", [50.0, 2.0, 80.0])
    responses = {
        "img0": _geo_result(["country level clue"]),   # passes both
        "img1": _geo_result(["country level clue"]),   # metric fails, residual passes
        "img2": _geo_result(["street level clue"]),    # metric passes, residual fails
    }
    report = verify_run(data, responses)

    assert report.target_rung == "city"
    assert report.threshold_km == 10.0
    assert report.matched == 3

    summary = report.as_json()["summary"]
    assert summary["passed"] == 1
    assert summary["failed"] == 2
    assert summary["metric_passes_residual_fails"] == 1
    assert summary["residual_passes_metric_fails"] == 1

    # Only the protected granularity is tallied, never an out-of-scope tag on a residual cue.
    assert report.residual_by_level() == {"street level clue": 1}
    assert "city" in report.render()


def test_report_round_trips_through_json(tmp_path):
    data = _evaluation_results("street level clue, city level clue", "error_km", [50.0])
    report = verify_run(data, {"img0": _geo_result(["country level clue"])})
    written = report.write(tmp_path / "verification.json")
    payload = json.loads(written.read_text())
    assert payload["threshold_km"] == 10.0
    assert payload["target_rung"] == "city"
    assert payload["images"]["img0"]["passed"] is True


def test_response_matching_tolerates_the_image_name_suffix():
    """Results are keyed by image id; caches are named after the image file stem."""
    data = _evaluation_results("street level clue, city level clue", "error_km", [50.0])
    report = verify_run(data, {"img0_leftImg8bit": _geo_result(["country level clue"])})
    assert report.matched == 1
    assert report.verdicts["img0"].residual_pass is True


# ---------------------------------------------------------------------------
# The superseded knob must not come back
# ---------------------------------------------------------------------------


def test_val_dist_is_rejected_with_a_migration_message(tmp_path):
    config = yaml.safe_load(Path("configs/geolocation.yaml").read_text())
    config["experiment"]["val_dist"] = 5.0
    path = tmp_path / "with_val_dist.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="val_dist"):
        load_config(str(path), check_dataset=False)


def test_mask_level_without_a_rung_is_rejected(tmp_path):
    config = yaml.safe_load(Path("configs/geolocation.yaml").read_text())
    config["experiment"]["mask_level"] = ["borough level clue"]
    path = tmp_path / "bad_mask_level.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="no known cue granularity"):
        load_config(str(path), check_dataset=False)


def test_shipped_configs_carry_no_superseded_keys():
    for name in ("configs/geolocation.yaml", "configs/compositional.yaml"):
        experiment = yaml.safe_load(Path(name).read_text())["experiment"]
        assert "val_dist" not in experiment, name


def test_summary_aggregate_is_refused_not_guessed(tmp_path, capsys):
    """
    A post-hoc aggregate has a score but no `feature_removed`, so it cannot say what granularity
    was protected. Refusing beats verifying against an assumed target.
    """
    from veillens.cli.verify import main

    path = tmp_path / "evaluation_results.json"
    path.write_text(json.dumps({"model": "x", "results": {}, "summary": {"mean": 0.1}}))
    assert main([str(path)]) == 1
    assert "does not say which granularity was protected" in capsys.readouterr().err


def test_inapplicable_metric_does_not_make_a_verdict_unknowable():
    """
    Compositional privacy has no distance rung, so the residual criterion is the whole verdict.
    Treating the absent metric as "unknown" would report every clean image as unverifiable.
    """
    clean = {"critical_features": {"features": [
        {"object_description": "a", "severity_level": 4},
    ]}}
    leaky = {"critical_features": {"features": [
        {"object_description": "b", "severity_level": 1},
    ]}}

    ok = verdict("a", privacy_class=COMP, mask_level=2, error=0.3, attacker_result=clean)
    assert ok.metric_pass is None and ok.threshold_km is None
    assert ok.residual_pass is True
    assert ok.passed is True

    bad = verdict("b", privacy_class=COMP, mask_level=2, error=0.3, attacker_result=leaky)
    assert bad.passed is False

    # Still unknowable when the residual criterion itself could not be computed.
    unknown = verdict("c", privacy_class=COMP, mask_level=2, error=0.3, attacker_result=None)
    assert unknown.passed is None


def test_geolocation_missing_error_is_still_unknown():
    """A class that has a rung but produced no error is genuinely unknown, not a pass."""
    clean = _geo_result(["country level clue"])
    v = verdict("x", privacy_class=GEO, mask_level=CITY_PROTECTION,
                error=None, attacker_result=clean)
    assert v.metric_pass is None
    assert v.residual_pass is True
    assert v.passed is None
