from verifier_anchored_sd.resource_profiles import e1_batch_sizes, e2_profile


def test_16gb_e1_sweeps_batch_until_capacity_boundary():
    assert e1_batch_sizes("16gb") == [1, 2, 4]


def test_24gb_e1_adds_a_larger_throughput_batch():
    assert e1_batch_sizes("24gb") == [1, 2, 4, 8]


def test_16gb_e2_uses_more_prompts_with_shorter_generation():
    profile = e2_profile("16gb")
    assert profile.prompts >= 64
    assert profile.prompt_tokens == 512
    assert profile.new_tokens == 64
    assert profile.gamma == 4


def test_24gb_e2_is_the_formal_long_generation_pilot():
    profile = e2_profile("24gb")
    assert profile.prompts == 200
    assert profile.prompt_tokens == 512
    assert profile.new_tokens == 512
    assert profile.gamma == 4
