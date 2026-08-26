"""
Iteration accounting for the protect loop.

R1 is iteration 0: its clues are inpainted on their own, snapshotted as
final/iter0_<name>_final.png, and only then does the gate re-read the image. That
makes each round's protected image a real inpaint of that round's clues alone.

These tests replay the ImageState transitions the workers perform, because the
sequencing is spread across three threads (seg worker, SD compositor, async API
workers) and the failure mode is silent: an is_complete() that never fires leaves
the image parked in the queue until walltime.
"""
from veillens.loop.state import ImageState


def new_state():
    return ImageState(image_id="img", image_path="/tmp/img.png", image_name="img")


def r1_segmentation(state, n_masks):
    """What seg_worker._process_r1_segmentation does to the state."""
    state.r2_iter = 0
    state.set_expected_r2_regions(0)
    state.set_num_total_clue(n_masks)


def advance(state, iter_idx):
    """What the compositor / _advance_iter_or_finalize_disk does to advance an iter."""
    state.reset_for_next_iter()
    state.r2_iter = iter_idx + 1


def gate_continues(state, iter_idx):
    """What api_workers._process_gate_eval does when the image still leaks."""
    state.reset_for_next_iter()
    state.r2_iter = iter_idx
    state.set_expected_r2_regions(1)
    state.add_r2_pred_path("/tmp/pred.json")


def test_iter0_completes_once_r1_clues_are_inpainted():
    state = new_state()
    r1_segmentation(state, n_masks=3)

    # The gate must not run until every R1 clue has been composited, or it would
    # read a half-protected image.
    assert not state.is_complete()
    state.increment_finished_clue()
    state.increment_finished_clue()
    assert not state.is_complete()
    state.increment_finished_clue()
    assert state.is_complete()
    assert state.r2_iter == 0  # so the snapshot is iter0_<name>_final.png


def test_iter0_with_no_grounded_clues_is_immediately_complete():
    # Nothing to inpaint means no SD item will ever reach the compositor; the seg
    # worker advances by hand, and that only works if the state agrees it is done.
    state = new_state()
    r1_segmentation(state, n_masks=0)
    assert state.is_complete()


def test_iteration_1_budget_excludes_r1_clues():
    state = new_state()
    r1_segmentation(state, n_masks=2)
    state.increment_finished_clue()
    state.increment_finished_clue()
    assert state.is_complete()

    advance(state, 0)
    assert state.r2_iter == 1
    # R1's finished clues must not count toward iteration 1, or iteration 1 would
    # report itself complete before inpainting anything.
    assert state.num_total_clue == 0
    assert state.num_finished_clue == 0
    assert not state.r2_seg_complete
    assert not state.is_complete()

    gate_continues(state, 1)
    assert not state.is_complete()  # waiting on this iteration's segmentation

    state.add_to_total_clue(2)  # R2 found two clues
    state.increment_r2_seg_count()
    assert state.r2_seg_complete
    assert not state.is_complete()

    state.increment_finished_clue()
    state.increment_finished_clue()
    assert state.is_complete()
    assert state.r2_iter == 1  # snapshot iter1_<name>_final.png


def test_suppressed_after_round_one_finalizes_on_the_inpaint():
    # The gate says the image is protected enough right after iteration 0. It must
    # finalize immediately -- previously this deadlocked if the state disagreed.
    state = new_state()
    r1_segmentation(state, n_masks=1)
    state.increment_finished_clue()
    advance(state, 0)

    state.stop_after_current_iter = True
    state.set_expected_r2_regions(0)
    assert state.is_complete()


def test_iteration_with_zero_new_clues_still_completes():
    state = new_state()
    r1_segmentation(state, n_masks=1)
    state.increment_finished_clue()
    advance(state, 0)

    gate_continues(state, 1)
    state.increment_r2_seg_count()  # segmented, but nothing grounded
    assert state.num_total_clue == 0
    assert state.is_complete()
