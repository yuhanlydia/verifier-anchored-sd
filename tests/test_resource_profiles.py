from verifier_anchored_sd.resource_profiles import e0_fit_profile


def test_16gb_formal_fit_reuses_freed_gpu_instead_of_cpu():
    full = e0_fit_profile("16gb", head_mode="full", draft_layers=28)
    assert full.accumulation_device == "cuda"
    assert full.selection_layer_block == 28
    assert full.fit_layer_block >= 2

    matched = e0_fit_profile("16gb", head_mode="matched", draft_layers=28)
    assert matched.accumulation_device == "cuda"
    assert matched.selection_layer_block == 28
    assert matched.fit_layer_block > full.fit_layer_block


def test_24gb_profile_keeps_all_layers_in_one_selection_pass():
    profile = e0_fit_profile("24gb", head_mode="full", draft_layers=28)
    assert profile.accumulation_device == "cuda"
    assert profile.selection_layer_block == 28
    assert profile.fit_layer_block >= 4
