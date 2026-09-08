# Target-Alignment GPU Run Report

## Run identity

```text
repo_commit_sha:
branch:
start_time:
end_time:
experiment_stage: A_8B_to_4B | B_32B_to_14B | E2_native_frontier
scientific: true | false
```

## Hardware / software

```text
GPU:
VRAM_GiB:
peak_GPU_memory_bytes:
host_RAM_GiB:
OS:
Python:
PyTorch:
CUDA_runtime:
Transformers:
Accelerate:
KVBridge:
```

## Frozen model contracts

```text
target_model:
target_revision_sha:
draft_model:
draft_revision_sha:
tokenizer_hash:
dtype:
quantization: none
```

## Frozen data / mapper contracts

```text
calibration_input_sha256:
evaluation_input_sha256:
E2_input_sha256:               # if applicable
mapper_sha256:
mapper_metadata_sha256:
requested_rows:
completed_rows:
independent_document_clusters:
```

## Primary target-alignment result

```text
mean_A_target_native:
mean_A_target_mapped:
mean_Delta_target:
Delta_target_95CI_low:
Delta_target_95CI_high:
native_fidelity_A_transfer:
decision_status: go_sd | stop_pair | expand | incomplete
```

## Runtime / failures

```text
elapsed_s:
OOM: false | true
failure_phase:
fallback_used:
configuration_changed_after_start: false
```

If OOM/failure occurred, paste the exact first error line and point to the committed
failure/progress JSON. Do not rerun with a different precision/model/prefix length
under the same scientific run identity.

## E2 only

```text
methods:
native_sd_realized_MAL:
mapped_init_only_realized_MAL:
mapped_accepted_only_realized_MAL:
accepted_only_minus_init_95CI:
native_tokens_per_s:
mapped_init_tokens_per_s:
mapped_accepted_only_tokens_per_s:
```

Do not copy the withdrawn 2026-09-03 overlap-product expected-MAL values into this
report. If the current conditional sampled-proposal estimator is reported, label it
explicitly as the **current conditional estimator**.

## Decision

```text
GO / STOP / EXPAND / DEBUG_ONLY:
reason:
next_permitted_command:
```
