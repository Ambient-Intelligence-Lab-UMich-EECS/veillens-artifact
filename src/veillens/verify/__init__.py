"""
Stage IV: verify that the granularities a run promised to remove are actually gone.

The loop is parameterised by ``experiment.mask_level`` -- which cue granularities to remove. This
package is the check that the parameter was honoured, and it is the only place in the codebase
that turns a run into a per-image pass/fail rather than a distribution.

    from veillens.verify import verify_run
    report = verify_run(evaluation_results, attacker_responses)
    print(report.render())

Import-light on purpose: no torch, no network, no config loading. ``veillens-verify`` runs on a
laptop against results produced on a cluster.
"""

from veillens.verify.granularity import (
    CLUE_LEVEL_TO_RUNG,
    LADDER_KM,
    ImageVerdict,
    ResidualCue,
    coarsest_level,
    protected_levels,
    residual_cues,
    target_rung,
    threshold_km,
    verdict,
)
from veillens.verify.report import RunVerification, verify_run

__all__ = [
    "CLUE_LEVEL_TO_RUNG",
    "LADDER_KM",
    "ImageVerdict",
    "ResidualCue",
    "RunVerification",
    "coarsest_level",
    "protected_levels",
    "residual_cues",
    "target_rung",
    "threshold_km",
    "verdict",
    "verify_run",
]
