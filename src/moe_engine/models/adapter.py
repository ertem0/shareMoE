"""Model adapter interface.

An adapter knows a model family's checkpoint layout. It builds expert modules
and extracts dense weights from a checkpoint, so the rest of the engine never
sees model-specific weight names.
"""

from typing import Protocol

import torch
from torch import nn

from moe_engine.checkpoint.loader import CheckpointReader
from moe_engine.experts.interface import ExpertId

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class IncompatibleCheckpointError(ValueError):
    """Checkpoint metadata or tensors do not match what the adapter expects."""


class UnsupportedDtypeError(ValueError):
    """The requested loading dtype is not supported."""


def check_dtype(dtype: torch.dtype) -> None:
    """Raise `UnsupportedDtypeError` unless `dtype` is a supported loading dtype."""
    if dtype not in SUPPORTED_DTYPES:
        raise UnsupportedDtypeError(
            f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype}"
        )


class ModelAdapter(Protocol):
    """Builds model components from a checkpoint."""

    def expert_ids(self) -> list[ExpertId]:
        """Every expert in the model, in ascending order."""
        ...

    def load_expert(
        self, reader: CheckpointReader, expert: ExpertId, dtype: torch.dtype
    ) -> nn.Module:
        """Build one expert, reading only that expert's tensors."""
        ...

    def load_dense_weights(
        self, reader: CheckpointReader, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        """Read every non-expert tensor, and no expert tensors."""
        ...
