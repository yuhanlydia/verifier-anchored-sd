"""Deployment wrapper that filters mapped draft KV through a frozen subspace.

Only mapped-only interventions are accepted.  Delta interventions need native draft
history and therefore cannot preserve the draft-prefill skip; they remain mechanism
upper bounds in the offline intervention evaluator.
"""

from __future__ import annotations

from ..kv_subspace import InterventionSpec, SubspaceBasisArtifact, apply_intervention
from .cache_state import CacheState, RotaryFactors


class SubspaceMappedKVMapper:
    """Expose the base mapper API while applying one frozen mapped-only filter."""

    def __init__(
        self,
        base_mapper,
        artifact: SubspaceBasisArtifact,
        spec: InterventionSpec,
    ) -> None:
        spec.validate()
        artifact.validate()
        if spec.mode not in {"mapped_soft", "orthogonal"}:
            raise ValueError(
                "deployment subspace mapper only supports mapped-only interventions; "
                f"{spec.mode!r} requires native draft history"
            )
        if spec.family not in artifact.bases:
            raise ValueError(f"deployment basis family {spec.family!r} is absent")
        metadata = getattr(base_mapper, "metadata", None)
        if metadata is None:
            raise ValueError("base mapper does not expose metadata")
        if int(metadata.draft_layers) != artifact.draft_layers:
            raise ValueError("base mapper and subspace artifact draft layer counts differ")
        if int(metadata.draft_kv_heads) != artifact.kv_heads:
            raise ValueError("base mapper and subspace artifact KV-head counts differ")
        if int(metadata.head_dim) != artifact.head_dim:
            raise ValueError("base mapper and subspace artifact head dimensions differ")
        self.base_mapper = base_mapper
        self.artifact = artifact
        self.spec = spec

    @property
    def metadata(self):
        return self.base_mapper.metadata

    @property
    def device(self):
        return self.base_mapper.device

    def map(
        self,
        target: CacheState,
        *,
        draft_rotary: RotaryFactors | None = None,
        include_residual: bool = True,
    ) -> CacheState:
        mapped = self.base_mapper.map(
            target,
            draft_rotary=draft_rotary,
            include_residual=include_residual,
        )
        # mapped-only modes ignore the native argument by construction. Passing the
        # same cache in both slots preserves the common geometry/rotary validation
        # in apply_intervention without materializing any native draft history.
        return apply_intervention(mapped, mapped, self.artifact, self.spec)
