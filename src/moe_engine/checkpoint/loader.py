"""Reads individual tensors from a local safetensors checkpoint.

Tensors are read by name, one file at a time, so loading a few weights never
materializes the whole model. This module is model-agnostic: which names
belong to which component is decided by the model adapter.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
SINGLE_FILE_NAME = "model.safetensors"


class CheckpointFormatError(ValueError):
    """Checkpoint files are missing, malformed or inconsistent."""


class MissingWeightError(LookupError):
    """A requested tensor is not in the checkpoint."""


class CheckpointReader:
    """Reads the config and individual tensors of a local checkpoint."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self._dir = checkpoint_dir
        self._weight_map = self._read_weight_map()

    @property
    def names(self) -> frozenset[str]:
        """Names of every tensor in the checkpoint."""
        return frozenset(self._weight_map)

    def config(self) -> dict[str, object]:
        """Return the parsed `config.json`."""
        config = self._read_json(self._dir / CONFIG_NAME)
        if not isinstance(config, dict):
            raise CheckpointFormatError(f"{CONFIG_NAME} must hold a JSON object")
        return config

    def load(self, names: Iterable[str]) -> dict[str, torch.Tensor]:
        """Read the named tensors, opening only the files that hold them."""
        by_file: dict[str, list[str]] = {}
        for name in names:
            file_name = self._weight_map.get(name)
            if file_name is None:
                raise MissingWeightError(f"tensor {name!r} is not in the checkpoint")
            by_file.setdefault(file_name, []).append(name)

        tensors: dict[str, torch.Tensor] = {}
        for file_name, file_names in by_file.items():
            path = self._dir / file_name
            try:
                with safe_open(path, framework="pt") as handle:
                    available = set(handle.keys())
                    for name in file_names:
                        if name not in available:
                            raise CheckpointFormatError(
                                f"index maps {name!r} to {file_name}, "
                                "but the file does not contain it"
                            )
                        tensors[name] = handle.get_tensor(name)
            except (SafetensorError, OSError) as exc:
                raise CheckpointFormatError(f"could not read {path}: {exc}") from exc
        return tensors

    def _read_weight_map(self) -> dict[str, str]:
        index_path = self._dir / INDEX_NAME
        if index_path.exists():
            return self._read_index(index_path)

        single_path = self._dir / SINGLE_FILE_NAME
        if single_path.exists():
            try:
                with safe_open(single_path, framework="pt") as handle:
                    return dict.fromkeys(handle.keys(), SINGLE_FILE_NAME)
            except (SafetensorError, OSError) as exc:
                raise CheckpointFormatError(
                    f"could not read {single_path}: {exc}"
                ) from exc

        raise CheckpointFormatError(
            f"{self._dir} has neither {INDEX_NAME} nor {SINGLE_FILE_NAME}"
        )

    def _read_index(self, path: Path) -> dict[str, str]:
        index = self._read_json(path)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise CheckpointFormatError(f"{path} has no 'weight_map' object")

        for name, file_name in weight_map.items():
            if not isinstance(file_name, str) or Path(file_name).name != file_name:
                raise CheckpointFormatError(
                    f"{path} maps {name!r} to invalid file {file_name!r}"
                )
        for file_name in set(weight_map.values()):
            if not (self._dir / file_name).is_file():
                raise CheckpointFormatError(f"{path} references missing {file_name}")
        return weight_map

    @staticmethod
    def _read_json(path: Path) -> object:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise CheckpointFormatError(f"{path} does not exist") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointFormatError(f"could not parse {path}: {exc}") from exc
