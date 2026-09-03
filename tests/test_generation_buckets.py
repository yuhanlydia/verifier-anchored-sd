from verifier_anchored_sd.evaluation import block_bucket_overlap


def test_bucket_overlap_uses_inclusive_output_positions():
    assert block_bucket_overlap(cursor=0, emitted=1, lo=1, hi=64) == 1
    assert block_bucket_overlap(cursor=63, emitted=1, lo=1, hi=64) == 1
    assert block_bucket_overlap(cursor=64, emitted=1, lo=1, hi=64) == 0
    assert block_bucket_overlap(cursor=64, emitted=1, lo=65, hi=128) == 1
    assert block_bucket_overlap(cursor=63, emitted=2, lo=1, hi=64) == 1
    assert block_bucket_overlap(cursor=63, emitted=2, lo=65, hi=128) == 1
