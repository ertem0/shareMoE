"""OLMoE model adapter.

All OLMoE-specific code lives in this module.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import skip_init

from moe_engine.experts.interface import (
    ExpertId,
    InvalidExpertInputError,
    UnknownExpertError,
)


@dataclass(frozen=True)
class OlmoeExpertConfig:
    """Expert layout of an OLMoE model."""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_experts: int
    initializer_range: float = 0.02

    def validate(self, expert: ExpertId) -> None:
        """Raise `UnknownExpertError` if `expert` is outside this model."""
        if expert.layer_id >= self.num_layers:
            raise UnknownExpertError(
                f"layer {expert.layer_id} does not exist, "
                f"the model has {self.num_layers} layers"
            )
        if expert.expert_id >= self.num_experts:
            raise UnknownExpertError(
                f"expert {expert.expert_id} does not exist, "
                f"each layer has {self.num_experts} experts"
            )


# Expert layout of allenai/OLMoE-1B-7B-0924.
OLMOE_1B_7B = OlmoeExpertConfig(
    hidden_size=2048,
    intermediate_size=1024,
    num_layers=16,
    num_experts=64,
)


class OlmoeExpert(nn.Module):
    """One OLMoE expert: a SwiGLU MLP without biases."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # skip_init leaves the weights uninitialized, so building an expert
        # never consumes the global random state.
        self.gate_proj = skip_init(
            nn.Linear, hidden_size, intermediate_size, bias=False, dtype=dtype
        )
        self.up_proj = skip_init(
            nn.Linear, hidden_size, intermediate_size, bias=False, dtype=dtype
        )
        self.down_proj = skip_init(
            nn.Linear, intermediate_size, hidden_size, bias=False, dtype=dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.gate_proj.weight
        if hidden_states.shape[-1] != weight.shape[1]:
            raise InvalidExpertInputError(
                f"expected hidden size {weight.shape[1]}, got {hidden_states.shape[-1]}"
            )
        if hidden_states.dtype != weight.dtype:
            raise InvalidExpertInputError(
                f"expected dtype {weight.dtype}, got {hidden_states.dtype}"
            )
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


def random_olmoe_expert(
    config: OlmoeExpertConfig,
    expert: ExpertId,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> OlmoeExpert:
    """Build an expert with random weights for testing.

    The weights depend only on `seed` and `expert`, so the same arguments
    always produce the same expert.
    """
    config.validate(expert)
    module = OlmoeExpert(config.hidden_size, config.intermediate_size, dtype=dtype)
    # Hashing a tuple of ints is deterministic across Python processes.
    generator = torch.Generator().manual_seed(
        hash((seed, expert.layer_id, expert.expert_id)) & 0xFFFF_FFFF_FFFF_FFFF
    )
    with torch.no_grad():
        for linear in (module.gate_proj, module.up_proj, module.down_proj):
            values = torch.empty(linear.weight.shape, dtype=torch.float32)
            values.normal_(mean=0.0, std=config.initializer_range, generator=generator)
            linear.weight.copy_(values)
    return module
