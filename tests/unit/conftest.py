import pytest

from moe_engine.models.olmoe import OlmoeExpertConfig


@pytest.fixture
def olmoe_1b_7b() -> OlmoeExpertConfig:
    """Expert layout of allenai/OLMoE-1B-7B-0924, from its published config.json.

    The engine reads these sizes from the checkpoint. Tests use them to build
    experts with the real shapes when there is no checkpoint to read.
    """
    return OlmoeExpertConfig(
        hidden_size=2048,
        intermediate_size=1024,
        num_layers=16,
        num_experts=64,
    )
