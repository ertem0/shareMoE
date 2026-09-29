import ast
from pathlib import Path

import pytest
import torch

import moe_engine.experts
import moe_engine.models
from moe_engine.experts.interface import (
    ExpertId,
    InvalidExpertIdError,
    InvalidExpertInputError,
    LocalExpertExecutor,
    UnknownExpertError,
)
from moe_engine.models.olmoe import OlmoeExpertConfig, random_olmoe_expert

SMALL = OlmoeExpertConfig(
    hidden_size=16, intermediate_size=8, num_layers=2, num_experts=4
)

NETWORKING_MODULES = {
    "asyncio",
    "http",
    "requests",
    "selectors",
    "socket",
    "socketserver",
    "ssl",
    "urllib",
    "moe_engine.protocol",
    "moe_engine.transport",
    "moe_engine.coordinator",
    "moe_engine.worker",
}


def test_expert_id_orders_by_layer_then_expert() -> None:
    ids = [ExpertId(1, 0), ExpertId(0, 3), ExpertId(0, 1)]

    assert sorted(ids) == [ExpertId(0, 1), ExpertId(0, 3), ExpertId(1, 0)]


def test_expert_id_is_hashable_and_compares_by_value() -> None:
    assert {ExpertId(2, 5): "x"}[ExpertId(2, 5)] == "x"


@pytest.mark.parametrize(
    ("layer_id", "expert_id"),
    [(-1, 0), (0, -1), (0.0, 1), (1, "2"), (True, 0)],
)
def test_expert_id_rejects_invalid_values(layer_id: object, expert_id: object) -> None:
    with pytest.raises(InvalidExpertIdError):
        ExpertId(layer_id, expert_id)  # type: ignore[arg-type]


def test_executor_runs_the_selected_expert() -> None:
    experts = {
        ExpertId(layer, expert): random_olmoe_expert(SMALL, ExpertId(layer, expert))
        for layer in range(SMALL.num_layers)
        for expert in range(SMALL.num_experts)
    }
    executor = LocalExpertExecutor(experts)
    hidden_states = torch.randn(5, SMALL.hidden_size)
    selected = ExpertId(1, 2)

    output = executor.execute(selected, hidden_states)

    torch.testing.assert_close(output, experts[selected](hidden_states))
    other = experts[ExpertId(1, 3)](hidden_states)
    assert not torch.allclose(output, other)


def test_executor_rejects_unknown_expert() -> None:
    executor = LocalExpertExecutor(
        {ExpertId(0, 0): random_olmoe_expert(SMALL, ExpertId(0, 0))}
    )

    with pytest.raises(UnknownExpertError):
        executor.execute(ExpertId(0, 1), torch.randn(3, SMALL.hidden_size))


@pytest.mark.parametrize("shape", [(SMALL.hidden_size,), (2, 3, SMALL.hidden_size)])
def test_executor_rejects_hidden_states_that_are_not_2d(
    shape: tuple[int, ...],
) -> None:
    executor = LocalExpertExecutor(
        {ExpertId(0, 0): random_olmoe_expert(SMALL, ExpertId(0, 0))}
    )

    with pytest.raises(InvalidExpertInputError):
        executor.execute(ExpertId(0, 0), torch.randn(shape))


def test_executor_does_not_track_gradients() -> None:
    executor = LocalExpertExecutor(
        {ExpertId(0, 0): random_olmoe_expert(SMALL, ExpertId(0, 0))}
    )

    output = executor.execute(ExpertId(0, 0), torch.randn(3, SMALL.hidden_size))

    assert not output.requires_grad


def _module_files(package: object) -> list[Path]:
    return sorted(Path(package.__file__).parent.rglob("*.py"))  # type: ignore[attr-defined]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def _is_networking(module: str) -> bool:
    return any(
        module == banned or module.startswith(f"{banned}.")
        for banned in NETWORKING_MODULES
    )


@pytest.mark.parametrize(
    "path",
    _module_files(moe_engine.experts) + _module_files(moe_engine.models),
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_expert_and_model_code_does_not_import_networking(path: Path) -> None:
    imported = _imported_modules(path)

    assert not {module for module in imported if _is_networking(module)}
