from verifier_anchored_sd.training.schedule import optimizer_microbatch_schedule


def test_steps_mean_optimizer_updates_not_microbatches():
    schedule = list(optimizer_microbatch_schedule(optimizer_steps=3, grad_accum=4))
    assert len(schedule) == 12
    assert schedule[:4] == [(1, 1), (1, 2), (1, 3), (1, 4)]
    assert schedule[-1] == (3, 4)


def test_schedule_rejects_nonpositive_counts():
    for steps, accum in ((0, 1), (1, 0), (-1, 4)):
        try:
            list(optimizer_microbatch_schedule(steps, accum))
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive schedule arguments must be rejected")
