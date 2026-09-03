import torch

from verifier_anchored_sd.device_utils import timing_device_from_parameter_devices


def test_timing_device_prefers_cuda_when_any_model_shard_is_on_cuda():
    assert timing_device_from_parameter_devices(["cpu", "cuda:0", "cpu"]) == torch.device("cuda:0")


def test_timing_device_falls_back_to_cpu_without_cuda_shards():
    assert timing_device_from_parameter_devices(["cpu", "cpu"]) == torch.device("cpu")
