import json
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from moe_engine.checkpoint.download import CheckpointSpec, ensure_checkpoint
from moe_engine.checkpoint.loader import (
    INDEX_NAME,
    SINGLE_FILE_NAME,
    CheckpointReader,
    MissingWeightError,
)
from moe_engine.experts.interface import (
    ExpertId,
    LocalExpertExecutor,
    UnknownExpertError,
)
from moe_engine.models.adapter import (
    IncompatibleCheckpointError,
    UnsupportedDtypeError,
)
from moe_engine.models.olmoe import (
    HF_MODEL_ID,
    HF_REVISION,
    OLMOE_1B_7B,
    OlmoeAdapter,
    OlmoeExpertConfig,
)

# The fields of allenai/OLMoE-1B-7B-0924's config.json that the adapter reads.
OLMOE_1B_7B_HF_CONFIG = {
    "model_type": "olmoe",
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 1024,
    "num_hidden_layers": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "vocab_size": 50304,
    "tie_word_embeddings": False,
}

SMALL_HF_CONFIG = {
    **OLMOE_1B_7B_HF_CONFIG,
    "hidden_size": 16,
    "intermediate_size": 8,
    "num_hidden_layers": 2,
    "num_experts": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "vocab_size": 32,
}

SMALL = OlmoeExpertConfig(
    hidden_size=16, intermediate_size=8, num_layers=2, num_experts=4
)


def small_checkpoint_tensors(
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    hidden, intermediate, vocab = 16, 8, 32

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(dtype)

    tensors = {
        "model.embed_tokens.weight": random(vocab, hidden),
        "model.norm.weight": random(hidden),
        "lm_head.weight": random(vocab, hidden),
    }
    for layer in range(SMALL.num_layers):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = random(hidden)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = random(hidden)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{prefix}.self_attn.{projection}.weight"] = random(hidden, hidden)
        tensors[f"{prefix}.self_attn.q_norm.weight"] = random(hidden)
        tensors[f"{prefix}.self_attn.k_norm.weight"] = random(hidden)
        tensors[f"{prefix}.mlp.gate.weight"] = random(SMALL.num_experts, hidden)
        for expert in range(SMALL.num_experts):
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            tensors[f"{expert_prefix}.gate_proj.weight"] = random(intermediate, hidden)
            tensors[f"{expert_prefix}.up_proj.weight"] = random(intermediate, hidden)
            tensors[f"{expert_prefix}.down_proj.weight"] = random(hidden, intermediate)
    return tensors


def write_checkpoint(
    directory: Path,
    tensors: dict[str, torch.Tensor],
    hf_config: dict[str, object] = SMALL_HF_CONFIG,
    sharded: bool = True,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(hf_config))
    if not sharded:
        save_file(tensors, directory / SINGLE_FILE_NAME)
        return

    # Shard by layer, like the real checkpoint splits its tensors over files.
    weight_map = {}
    shards: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        file_name = (
            "layer-1.safetensors" if ".layers.1." in name else "rest.safetensors"
        )
        shards.setdefault(file_name, {})[name] = tensor
        weight_map[name] = file_name
    for file_name, shard in shards.items():
        save_file(shard, directory / file_name)
    (directory / INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))


class RecordingReader(CheckpointReader):
    """Records which tensors are read."""

    def __init__(self, checkpoint_dir: Path) -> None:
        super().__init__(checkpoint_dir)
        self.loaded: list[str] = []

    def load(self, names: Iterable[str]) -> dict[str, torch.Tensor]:
        names = list(names)
        self.loaded.extend(names)
        return super().load(names)


def reference_expert(
    tensors: dict[str, torch.Tensor], expert: ExpertId, x: torch.Tensor
) -> torch.Tensor:
    prefix = f"model.layers.{expert.layer_id}.mlp.experts.{expert.expert_id}"
    gate = tensors[f"{prefix}.gate_proj.weight"].to(x.dtype)
    up = tensors[f"{prefix}.up_proj.weight"].to(x.dtype)
    down = tensors[f"{prefix}.down_proj.weight"].to(x.dtype)
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


@pytest.fixture
def adapter() -> OlmoeAdapter:
    return OlmoeAdapter.from_hf_config(SMALL_HF_CONFIG)


def test_real_config_gives_the_olmoe_1b_7b_layout() -> None:
    adapter = OlmoeAdapter.from_hf_config(OLMOE_1B_7B_HF_CONFIG)

    assert adapter.config == OLMOE_1B_7B
    assert not adapter.tie_word_embeddings


def test_pinned_checkpoint_spec_is_valid(tmp_path: Path) -> None:
    CheckpointSpec(model_id=HF_MODEL_ID, revision=HF_REVISION, local_dir=tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"model_type": "mixtral"},
        {"hidden_act": "gelu"},
        {"hidden_size": 0},
        {"num_experts": "64"},
        {"num_hidden_layers": True},
        {"tie_word_embeddings": "no"},
    ],
)
def test_incompatible_config_is_rejected(override: dict[str, object]) -> None:
    with pytest.raises(IncompatibleCheckpointError):
        OlmoeAdapter.from_hf_config({**SMALL_HF_CONFIG, **override})


def test_missing_config_field_is_rejected() -> None:
    config = dict(SMALL_HF_CONFIG)
    del config["intermediate_size"]

    with pytest.raises(IncompatibleCheckpointError):
        OlmoeAdapter.from_hf_config(config)


def test_expert_ids_cover_every_expert_in_order(adapter: OlmoeAdapter) -> None:
    ids = adapter.expert_ids()

    assert len(ids) == SMALL.num_layers * SMALL.num_experts
    assert ids == sorted(ids)
    assert ids[0] == ExpertId(0, 0)
    assert ids[-1] == ExpertId(1, 3)


@pytest.mark.parametrize("sharded", [True, False])
def test_loaded_expert_matches_reference_execution(
    tmp_path: Path, adapter: OlmoeAdapter, sharded: bool
) -> None:
    tensors = small_checkpoint_tensors()
    write_checkpoint(tmp_path, tensors, sharded=sharded)
    selected = ExpertId(1, 2)
    executor = LocalExpertExecutor(
        {selected: adapter.load_expert(CheckpointReader(tmp_path), selected)}
    )
    x = torch.randn(5, SMALL.hidden_size)

    output = executor.execute(selected, x)

    torch.testing.assert_close(output, reference_expert(tensors, selected, x))


def test_loading_an_expert_reads_only_its_tensors(
    tmp_path: Path, adapter: OlmoeAdapter
) -> None:
    write_checkpoint(tmp_path, small_checkpoint_tensors())
    reader = RecordingReader(tmp_path)

    adapter.load_expert(reader, ExpertId(0, 3))

    assert sorted(reader.loaded) == [
        "model.layers.0.mlp.experts.3.down_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.up_proj.weight",
    ]


def test_bfloat16_checkpoint_loads_into_float32(
    tmp_path: Path, adapter: OlmoeAdapter
) -> None:
    tensors = small_checkpoint_tensors(dtype=torch.bfloat16)
    write_checkpoint(tmp_path, tensors)
    selected = ExpertId(0, 1)
    expert = adapter.load_expert(CheckpointReader(tmp_path), selected)
    x = torch.randn(3, SMALL.hidden_size)

    with torch.no_grad():
        output = expert(x)

    assert expert.gate_proj.weight.dtype == torch.float32
    torch.testing.assert_close(output, reference_expert(tensors, selected, x))


def test_expert_loads_in_the_requested_dtype(
    tmp_path: Path, adapter: OlmoeAdapter
) -> None:
    tensors = small_checkpoint_tensors()
    write_checkpoint(tmp_path, tensors)
    selected = ExpertId(1, 0)
    expert = adapter.load_expert(
        CheckpointReader(tmp_path), selected, dtype=torch.bfloat16
    )
    x = torch.randn(3, SMALL.hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        output = expert(x)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reference_expert(tensors, selected, x))


def test_unsupported_dtype_is_rejected(tmp_path: Path, adapter: OlmoeAdapter) -> None:
    write_checkpoint(tmp_path, small_checkpoint_tensors())

    with pytest.raises(UnsupportedDtypeError):
        adapter.load_expert(
            CheckpointReader(tmp_path), ExpertId(0, 0), dtype=torch.float64
        )


@pytest.mark.parametrize("expert", [ExpertId(2, 0), ExpertId(0, 4)])
def test_unknown_expert_fails_before_reading(
    tmp_path: Path, adapter: OlmoeAdapter, expert: ExpertId
) -> None:
    write_checkpoint(tmp_path, small_checkpoint_tensors())
    reader = RecordingReader(tmp_path)

    with pytest.raises(UnknownExpertError):
        adapter.load_expert(reader, expert)

    assert reader.loaded == []


def test_missing_expert_weight_fails(tmp_path: Path, adapter: OlmoeAdapter) -> None:
    tensors = small_checkpoint_tensors()
    del tensors["model.layers.0.mlp.experts.2.up_proj.weight"]
    write_checkpoint(tmp_path, tensors)

    with pytest.raises(MissingWeightError):
        adapter.load_expert(CheckpointReader(tmp_path), ExpertId(0, 2))


@pytest.mark.parametrize(
    "replacement",
    [torch.zeros(8, 17), torch.zeros(16, 8), torch.zeros(8, 16, dtype=torch.int32)],
)
def test_incompatible_expert_tensor_fails(
    tmp_path: Path, adapter: OlmoeAdapter, replacement: torch.Tensor
) -> None:
    tensors = small_checkpoint_tensors()
    tensors["model.layers.0.mlp.experts.0.gate_proj.weight"] = replacement
    write_checkpoint(tmp_path, tensors)

    with pytest.raises(IncompatibleCheckpointError):
        adapter.load_expert(CheckpointReader(tmp_path), ExpertId(0, 0))


def test_dense_weights_exclude_every_expert_tensor(
    tmp_path: Path, adapter: OlmoeAdapter
) -> None:
    tensors = small_checkpoint_tensors()
    write_checkpoint(tmp_path, tensors)
    reader = RecordingReader(tmp_path)

    dense = adapter.load_dense_weights(reader)

    expected = {name for name in tensors if ".mlp.experts." not in name}
    assert set(dense) == expected
    assert set(reader.loaded) == expected
    for name, tensor in dense.items():
        torch.testing.assert_close(tensor, tensors[name])


def test_dense_weights_load_in_the_requested_dtype(
    tmp_path: Path, adapter: OlmoeAdapter
) -> None:
    write_checkpoint(tmp_path, small_checkpoint_tensors())

    dense = adapter.load_dense_weights(CheckpointReader(tmp_path), dtype=torch.float16)

    assert {tensor.dtype for tensor in dense.values()} == {torch.float16}


def test_tied_embeddings_skip_the_lm_head(tmp_path: Path) -> None:
    config = {**SMALL_HF_CONFIG, "tie_word_embeddings": True}
    tensors = small_checkpoint_tensors()
    del tensors["lm_head.weight"]
    write_checkpoint(tmp_path, tensors, hf_config=config)
    adapter = OlmoeAdapter.from_hf_config(config)

    dense = adapter.load_dense_weights(CheckpointReader(tmp_path))

    assert "lm_head.weight" not in dense
    assert "model.embed_tokens.weight" in dense


def test_missing_dense_weight_fails(tmp_path: Path, adapter: OlmoeAdapter) -> None:
    tensors = small_checkpoint_tensors()
    del tensors["model.layers.1.self_attn.k_norm.weight"]
    write_checkpoint(tmp_path, tensors)

    with pytest.raises(MissingWeightError):
        adapter.load_dense_weights(CheckpointReader(tmp_path))


def test_downloaded_checkpoint_loads_offline_on_a_later_start(tmp_path: Path) -> None:
    tensors = small_checkpoint_tensors()
    spec = CheckpointSpec(
        model_id=HF_MODEL_ID, revision=HF_REVISION, local_dir=tmp_path / "olmoe"
    )

    def fixture_downloader(spec: CheckpointSpec) -> None:
        write_checkpoint(spec.local_dir, tensors)

    def offline_downloader(spec: CheckpointSpec) -> None:
        raise AssertionError("a later start must not download")

    ensure_checkpoint(spec, fixture_downloader)
    checkpoint_dir = ensure_checkpoint(spec, offline_downloader)
    reader = CheckpointReader(checkpoint_dir)
    adapter = OlmoeAdapter.from_hf_config(reader.config())
    selected = ExpertId(1, 3)
    x = torch.randn(4, SMALL.hidden_size)

    with torch.no_grad():
        output = adapter.load_expert(reader, selected)(x)

    torch.testing.assert_close(output, reference_expert(tensors, selected, x))
