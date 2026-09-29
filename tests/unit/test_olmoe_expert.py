import pytest
import torch
from torch.nn import functional as F

from moe_engine.experts.interface import (
    ExpertId,
    InvalidExpertInputError,
    LocalExpertExecutor,
    UnknownExpertError,
)
from moe_engine.models.olmoe import OlmoeExpert, OlmoeExpertConfig
from support.olmoe import random_olmoe_expert

SMALL = OlmoeExpertConfig(
    hidden_size=16, intermediate_size=8, num_layers=2, num_experts=4
)


def test_expert_has_olmoe_weight_shapes(olmoe_1b_7b: OlmoeExpertConfig) -> None:
    expert = random_olmoe_expert(olmoe_1b_7b, ExpertId(0, 0))

    assert expert.gate_proj.weight.shape == (1024, 2048)
    assert expert.up_proj.weight.shape == (1024, 2048)
    assert expert.down_proj.weight.shape == (2048, 1024)
    assert all(linear.bias is None for linear in expert.children())


def test_selected_expert_executes_with_olmoe_output_shape(
    olmoe_1b_7b: OlmoeExpertConfig,
) -> None:
    selected = ExpertId(15, 63)
    executor = LocalExpertExecutor(
        {selected: random_olmoe_expert(olmoe_1b_7b, selected)}
    )
    hidden_states = torch.randn(7, olmoe_1b_7b.hidden_size)

    output = executor.execute(selected, hidden_states)

    assert output.shape == (7, olmoe_1b_7b.hidden_size)
    assert output.dtype == torch.float32


def test_repeated_execution_gives_the_same_output(
    olmoe_1b_7b: OlmoeExpertConfig,
) -> None:
    selected = ExpertId(3, 41)
    executor = LocalExpertExecutor(
        {selected: random_olmoe_expert(olmoe_1b_7b, selected)}
    )
    hidden_states = torch.randn(7, olmoe_1b_7b.hidden_size)

    first = executor.execute(selected, hidden_states)
    second = executor.execute(selected, hidden_states)

    torch.testing.assert_close(first, second)


def test_building_an_expert_does_not_consume_global_random_state() -> None:
    torch.manual_seed(0)
    expected = torch.rand(1)

    torch.manual_seed(0)
    OlmoeExpert(SMALL.hidden_size, SMALL.intermediate_size)
    actual = torch.rand(1)

    torch.testing.assert_close(actual, expected)


def test_expert_computes_swiglu() -> None:
    expert = random_olmoe_expert(SMALL, ExpertId(0, 1))
    x = torch.randn(4, SMALL.hidden_size)
    gate = expert.gate_proj.weight
    up = expert.up_proj.weight
    down = expert.down_proj.weight

    expected = (F.silu(x @ gate.T) * (x @ up.T)) @ down.T

    with torch.no_grad():
        torch.testing.assert_close(expert(x), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_expert_preserves_dtype(dtype: torch.dtype) -> None:
    selected = ExpertId(0, 0)
    executor = LocalExpertExecutor(
        {selected: random_olmoe_expert(SMALL, selected, dtype=dtype)}
    )
    hidden_states = torch.randn(3, SMALL.hidden_size, dtype=dtype)

    output = executor.execute(selected, hidden_states)

    assert output.dtype == dtype
    assert output.shape == hidden_states.shape


def test_empty_batch_gives_empty_output() -> None:
    selected = ExpertId(0, 0)
    executor = LocalExpertExecutor({selected: random_olmoe_expert(SMALL, selected)})

    output = executor.execute(selected, torch.empty(0, SMALL.hidden_size))

    assert output.shape == (0, SMALL.hidden_size)


@pytest.mark.parametrize("expert", [ExpertId(2, 0), ExpertId(0, 4)])
def test_config_rejects_ids_outside_the_model(expert: ExpertId) -> None:
    with pytest.raises(UnknownExpertError):
        SMALL.validate(expert)


def test_expert_rejects_wrong_hidden_size() -> None:
    selected = ExpertId(0, 0)
    executor = LocalExpertExecutor({selected: random_olmoe_expert(SMALL, selected)})

    with pytest.raises(InvalidExpertInputError):
        executor.execute(selected, torch.randn(3, SMALL.hidden_size + 1))


def test_expert_rejects_mismatched_dtype() -> None:
    selected = ExpertId(0, 0)
    executor = LocalExpertExecutor({selected: random_olmoe_expert(SMALL, selected)})

    with pytest.raises(InvalidExpertInputError):
        executor.execute(
            selected, torch.randn(3, SMALL.hidden_size, dtype=torch.bfloat16)
        )
