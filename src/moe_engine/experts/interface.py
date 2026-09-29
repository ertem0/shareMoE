"""Model-agnostic expert execution interface.

The dense runtime executes experts only through `ExpertExecutor`. Whether an
expert runs in this process or on a remote worker is an implementation detail
of the executor, so this module must never depend on networking.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class ExpertError(Exception):
    """Base class for controlled expert errors."""


class InvalidExpertIdError(ExpertError, ValueError):
    """An expert ID is not a pair of non-negative integers."""


class UnknownExpertError(ExpertError, LookupError):
    """An expert or layer ID does not exist or is not available here."""


class InvalidExpertInputError(ExpertError, ValueError):
    """Hidden states have a shape or dtype the expert cannot accept."""


@dataclass(frozen=True, order=True)
class ExpertId:
    """Identifies one expert by its layer and its index within that layer."""

    layer_id: int
    expert_id: int

    def __post_init__(self) -> None:
        for name in ("layer_id", "expert_id"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidExpertIdError(
                    f"{name} must be an int, got {type(value).__name__}"
                )
            if value < 0:
                raise InvalidExpertIdError(f"{name} must be non-negative, got {value}")

    def __str__(self) -> str:
        return f"(layer {self.layer_id}, expert {self.expert_id})"


class ExpertExecutor(Protocol):
    """Executes one expert on a batch of hidden states."""

    def execute(self, expert: ExpertId, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply `expert` to `hidden_states` of shape `[num_tokens, hidden_size]`.

        Returns a tensor with the same shape and dtype as `hidden_states`.
        """
        ...


class LocalExpertExecutor:
    """Executes experts held as modules in this process."""

    def __init__(self, experts: Mapping[ExpertId, nn.Module]) -> None:
        self._experts = dict(experts)

    def execute(self, expert: ExpertId, hidden_states: torch.Tensor) -> torch.Tensor:
        module = self._experts.get(expert)
        if module is None:
            raise UnknownExpertError(f"expert {expert} is not available")
        if hidden_states.dim() != 2:
            raise InvalidExpertInputError(
                "hidden states must have shape [num_tokens, hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        with torch.no_grad():
            return module(hidden_states)
