"""A run that grounds nothing must not report success.

Without the compiled extension GroundingDINO needs, every detection raises inside the
segmentation worker, the loop catches it per clue and records "no ground", and the run reaches
the end with exit 0 and a full score table -- computed by re-reading the unedited originals. A
weak result and a broken install look identical from the outside. This pins the guard that
separates them.

The guard lives on ClueStatsTracker rather than on the pipeline so that it can be tested here
without importing torch, which the rest of this suite also avoids.
"""

import pytest

from veillens.loop.state import ClueStatsTracker


def _tracker(selected=(), skipped=()):
    tracker = ClueStatsTracker()
    for round_tag, level in selected:
        tracker.record_selected(round_tag, {"properties": [level]})
    for round_tag, level in skipped:
        tracker.record_skipped(round_tag, {"properties": [level]})
    return tracker


def test_nothing_selected_is_not_a_failure():
    """An oracle that finds no clues to remove is a legitimate outcome, not a broken install."""
    _tracker().assert_something_was_grounded()


def test_some_grounded_passes():
    _tracker(selected=[("R1", "city level clue"), ("R1", "street level clue")],
             skipped=[("R1", "city level clue")]).assert_something_was_grounded()


def test_round_two_grounding_rescues_a_round_one_that_grounded_nothing():
    _tracker(selected=[("R1", "city level clue"), ("R2", "street level clue")],
             skipped=[("R1", "city level clue")]).assert_something_was_grounded()


def test_all_selected_but_none_grounded_raises():
    selected = [("R1", "city level clue"), ("R1", "street level clue"), ("R2", "city level clue")]
    with pytest.raises(RuntimeError) as exc:
        _tracker(selected=selected, skipped=selected).assert_something_was_grounded()
    message = str(exc.value)
    assert "3 clues were selected and 0 were localised" in message
    # The message has to name the cause, or the next person re-runs it unchanged.
    assert "_C" in message
    assert "--dry-run" in message
