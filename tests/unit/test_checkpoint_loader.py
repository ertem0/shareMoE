import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from moe_engine.checkpoint.loader import (
    INDEX_NAME,
    SINGLE_FILE_NAME,
    CheckpointFormatError,
    CheckpointReader,
    MissingWeightError,
)

TENSORS = {
    "a.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    "b.weight": torch.ones(4, dtype=torch.bfloat16),
    "c.weight": torch.zeros(2, 2, dtype=torch.float16),
}


def write_single(directory: Path) -> None:
    save_file(TENSORS, directory / SINGLE_FILE_NAME)


def write_sharded(directory: Path) -> None:
    save_file({"a.weight": TENSORS["a.weight"]}, directory / "part-1.safetensors")
    save_file(
        {name: TENSORS[name] for name in ("b.weight", "c.weight")},
        directory / "part-2.safetensors",
    )
    write_index(
        directory,
        {
            "a.weight": "part-1.safetensors",
            "b.weight": "part-2.safetensors",
            "c.weight": "part-2.safetensors",
        },
    )


def write_index(directory: Path, weight_map: dict[str, str]) -> None:
    (directory / INDEX_NAME).write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


@pytest.mark.parametrize("write", [write_single, write_sharded])
def test_reads_individual_tensors(tmp_path: Path, write) -> None:
    write(tmp_path)
    reader = CheckpointReader(tmp_path)

    loaded = reader.load(["a.weight", "c.weight"])

    assert reader.names == frozenset(TENSORS)
    assert loaded.keys() == {"a.weight", "c.weight"}
    for name, tensor in loaded.items():
        assert tensor.dtype == TENSORS[name].dtype
        torch.testing.assert_close(tensor, TENSORS[name])


def test_reads_only_the_requested_shard(tmp_path: Path) -> None:
    write_sharded(tmp_path)
    reader = CheckpointReader(tmp_path)
    (tmp_path / "part-2.safetensors").write_bytes(b"corrupted after indexing")

    loaded = reader.load(["a.weight"])

    torch.testing.assert_close(loaded["a.weight"], TENSORS["a.weight"])


def test_unknown_tensor_name_fails(tmp_path: Path) -> None:
    write_single(tmp_path)

    with pytest.raises(MissingWeightError):
        CheckpointReader(tmp_path).load(["missing.weight"])


def test_missing_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(CheckpointFormatError, match="does not exist"):
        CheckpointReader(tmp_path / "absent")


def test_file_instead_of_directory_fails(tmp_path: Path) -> None:
    write_single(tmp_path)

    with pytest.raises(CheckpointFormatError, match="does not exist"):
        CheckpointReader(tmp_path / SINGLE_FILE_NAME)


def test_empty_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(CheckpointFormatError):
        CheckpointReader(tmp_path)


def test_corrupt_single_file_fails(tmp_path: Path) -> None:
    (tmp_path / SINGLE_FILE_NAME).write_bytes(b"not safetensors")

    with pytest.raises(CheckpointFormatError):
        CheckpointReader(tmp_path)


def test_corrupt_shard_fails_on_load(tmp_path: Path) -> None:
    write_sharded(tmp_path)
    reader = CheckpointReader(tmp_path)
    (tmp_path / "part-1.safetensors").write_bytes(b"not safetensors")

    with pytest.raises(CheckpointFormatError):
        reader.load(["a.weight"])


def test_index_referencing_a_missing_file_fails(tmp_path: Path) -> None:
    write_index(tmp_path, {"a.weight": "absent.safetensors"})

    with pytest.raises(CheckpointFormatError):
        CheckpointReader(tmp_path)


def test_index_entry_absent_from_its_file_fails(tmp_path: Path) -> None:
    write_sharded(tmp_path)
    write_index(tmp_path, {"b.weight": "part-1.safetensors"})
    reader = CheckpointReader(tmp_path)

    with pytest.raises(CheckpointFormatError):
        reader.load(["b.weight"])


@pytest.mark.parametrize(
    "index",
    ["not json", json.dumps([]), json.dumps({"weight_map": []})],
)
def test_malformed_index_fails(tmp_path: Path, index: str) -> None:
    (tmp_path / INDEX_NAME).write_text(index)

    with pytest.raises(CheckpointFormatError):
        CheckpointReader(tmp_path)


def test_index_with_a_path_outside_the_directory_fails(tmp_path: Path) -> None:
    write_index(tmp_path, {"a.weight": "../part-1.safetensors"})

    with pytest.raises(CheckpointFormatError):
        CheckpointReader(tmp_path)


def test_reads_config(tmp_path: Path) -> None:
    write_single(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 16}))

    assert CheckpointReader(tmp_path).config() == {"hidden_size": 16}


@pytest.mark.parametrize("content", [None, "not json", "[1, 2]"])
def test_missing_or_malformed_config_fails(tmp_path: Path, content: str | None) -> None:
    write_single(tmp_path)
    if content is not None:
        (tmp_path / "config.json").write_text(content)

    with pytest.raises(CheckpointFormatError):
        CheckpointReader(tmp_path).config()
