"""OLMoE model adapter.

All OLMoE-specific code lives in this module.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from moe_engine.checkpoint.loader import CheckpointReader
from moe_engine.experts.interface import (
    ExpertId,
    InvalidExpertInputError,
    UnknownExpertError,
)
from moe_engine.models.adapter import IncompatibleCheckpointError, check_dtype

_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LAYER_DENSE_WEIGHTS = (
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "mlp.gate",
)


@dataclass(frozen=True)
class OlmoeExpertConfig:
    """Expert layout of an OLMoE model."""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_experts: int

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


def _uninitialized_linear(
    in_features: int, out_features: int, dtype: torch.dtype
) -> nn.Linear:
    # Built on the meta device, so the weights are allocated but never
    # initialized and building an expert never consumes the global random state.
    linear = nn.Linear(
        in_features, out_features, bias=False, device="meta", dtype=dtype
    )
    return linear.to_empty(device="cpu")


class OlmoeExpert(nn.Module):
    """One OLMoE expert: a SwiGLU MLP without biases."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.gate_proj = _uninitialized_linear(hidden_size, intermediate_size, dtype)
        self.up_proj = _uninitialized_linear(hidden_size, intermediate_size, dtype)
        self.down_proj = _uninitialized_linear(intermediate_size, hidden_size, dtype)

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


class OlmoeAdapter:
    """Builds OLMoE components from a checkpoint in the Hugging Face layout."""

    def __init__(
        self, config: OlmoeExpertConfig, tie_word_embeddings: bool = False
    ) -> None:
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings

    @classmethod
    def from_hf_config(cls, hf_config: Mapping[str, object]) -> "OlmoeAdapter":
        """Build an adapter from the checkpoint's `config.json`."""
        if hf_config.get("model_type") != "olmoe":
            raise IncompatibleCheckpointError(
                f"expected model_type 'olmoe', got {hf_config.get('model_type')!r}"
            )
        if hf_config.get("hidden_act") != "silu":
            raise IncompatibleCheckpointError(
                f"expected hidden_act 'silu', got {hf_config.get('hidden_act')!r}"
            )
        tie_word_embeddings = hf_config.get("tie_word_embeddings", False)
        if not isinstance(tie_word_embeddings, bool):
            raise IncompatibleCheckpointError("tie_word_embeddings must be a bool")
        config = OlmoeExpertConfig(
            hidden_size=_positive_int(hf_config, "hidden_size"),
            intermediate_size=_positive_int(hf_config, "intermediate_size"),
            num_layers=_positive_int(hf_config, "num_hidden_layers"),
            num_experts=_positive_int(hf_config, "num_experts"),
        )
        return cls(config, tie_word_embeddings=tie_word_embeddings)

    def expert_ids(self) -> list[ExpertId]:
        return [
            ExpertId(layer_id, expert_id)
            for layer_id in range(self.config.num_layers)
            for expert_id in range(self.config.num_experts)
        ]

    def expert_weight_names(self, expert: ExpertId) -> dict[str, str]:
        """Map each projection of `expert` to its checkpoint tensor name."""
        self.config.validate(expert)
        prefix = f"model.layers.{expert.layer_id}.mlp.experts.{expert.expert_id}"
        return {
            projection: f"{prefix}.{projection}.weight"
            for projection in _EXPERT_PROJECTIONS
        }

    def dense_weight_names(self) -> list[str]:
        """Checkpoint names of every non-expert tensor."""
        names = ["model.embed_tokens.weight"]
        for layer_id in range(self.config.num_layers):
            names.extend(
                f"model.layers.{layer_id}.{weight}.weight"
                for weight in _LAYER_DENSE_WEIGHTS
            )
        names.append("model.norm.weight")
        if not self.tie_word_embeddings:
            names.append("lm_head.weight")
        return names

    def load_expert(
        self,
        reader: CheckpointReader,
        expert: ExpertId,
        dtype: torch.dtype = torch.float32,
    ) -> OlmoeExpert:
        check_dtype(dtype)
        names = self.expert_weight_names(expert)
        tensors = reader.load(names.values())
        hidden = self.config.hidden_size
        intermediate = self.config.intermediate_size
        expected_shapes = {
            "gate_proj": (intermediate, hidden),
            "up_proj": (intermediate, hidden),
            "down_proj": (hidden, intermediate),
        }

        module = OlmoeExpert(hidden, intermediate, dtype=dtype)
        with torch.no_grad():
            for projection, name in names.items():
                tensor = tensors[name]
                _check_tensor(name, tensor, expected_shapes[projection])
                getattr(module, projection).weight.copy_(tensor)
        return module

    def load_dense_weights(
        self, reader: CheckpointReader, dtype: torch.dtype = torch.float32
    ) -> dict[str, torch.Tensor]:
        check_dtype(dtype)
        tensors = reader.load(self.dense_weight_names())
        for name, tensor in tensors.items():
            _check_tensor(name, tensor, expected_shape=None)
        return {name: tensor.to(dtype) for name, tensor in tensors.items()}


def _positive_int(hf_config: Mapping[str, object], key: str) -> int:
    value = hf_config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise IncompatibleCheckpointError(
            f"{key} must be a positive int, got {value!r}"
        )
    return value


def _check_tensor(
    name: str, tensor: torch.Tensor, expected_shape: tuple[int, ...] | None
) -> None:
    if not tensor.dtype.is_floating_point:
        raise IncompatibleCheckpointError(
            f"{name} must be a floating-point tensor, got {tensor.dtype}"
        )
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise IncompatibleCheckpointError(
            f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}"
        )
