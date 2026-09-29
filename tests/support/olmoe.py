"""OLMoE test helpers."""

import torch

from moe_engine.experts.interface import ExpertId
from moe_engine.models.olmoe import OlmoeExpert, OlmoeExpertConfig


def random_olmoe_expert(
    config: OlmoeExpertConfig,
    expert: ExpertId,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    std: float = 0.02,
) -> OlmoeExpert:
    """Build an expert with random weights.

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
            values.normal_(mean=0.0, std=std, generator=generator)
            linear.weight.copy_(values)
    return module
