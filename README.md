# Distributed MoE Inference Engine

A research prototype that runs Mixture-of-Experts (MoE) language models with their experts spread across several machines, so that no single device has to hold the whole model in memory.

> **Status:** early development. See [docs/PROJECT.md](docs/PROJECT.md) for the milestone plan and current progress.

## How it works

The system has two node roles.

- **Coordinator.** Holds the full model checkpoint and computes every non-expert component (embeddings, attention, routers, normalization and the LM head). It distributes experts to workers at startup, tracks which worker holds each expert, monitors worker health and dispatches hidden states during inference.
- **Worker.** Requests experts from the coordinator based on the memory it can offer, receives and serves those experts, and computes expert outputs for the hidden states the coordinator sends.

For every layer, the coordinator computes attention and the router, sends each token's hidden states to the workers holding the selected experts, and combines the returned outputs with the router weights before moving to the next layer.

```text
prompt ──► coordinator: attention + router (layer 0)
              │
              ├──► worker 1: experts ──┐
              └──► worker 2: experts ──┤
                                       ▼
           coordinator: weighted sum ──► attention + router (layer 1) ──► ... ──► result
```

Any node, the coordinator or a worker, can submit an inference request. The result is returned to the node that asked.

Each worker keeps two TCP connections to the coordinator. The control connection carries initialization, heartbeats and inference requests. The tensor connection carries expert weights and hidden states. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the components, the wire protocol and the message sequences.

## Supported models

- [`allenai/OLMoE-1B-7B-0924`](https://huggingface.co/allenai/OLMoE-1B-7B-0924)

Model-specific code is isolated in a model adapter, so other MoE models can be added without changing the engine.

## Requirements

- Python
- [uv](https://docs.astral.sh/uv/)
- PyTorch (installed through uv)

## Setup

```bash
git clone <repository-url>
cd <repository>
uv sync
```

Run every command through `uv run` rather than an activated virtual environment.

## Development

Run the tests:

```bash
uv run pytest                    # everything
uv run pytest tests/unit         # unit tests only
uv run pytest tests/integration  # integration tests only
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format .
```

Before committing, all three must pass:

```bash
uv run ruff format . && uv run ruff check . && uv run pytest
```

Tests run offline with small local fixtures and never download the model.

## Running the engine

Not available yet. Starting a coordinator and workers becomes possible from Milestone 7, and full inference from Milestone 10. This section will document the commands once they exist.

## Documentation

| Document                                     | Contents                                                  |
| :------------------------------------------- | :-------------------------------------------------------- |
| [docs/PROJECT.md](docs/PROJECT.md)           | Goals, scope, milestones and future scope                 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, expert directory, wire protocol and sequences |
| [CONTRIBUTING.md](CONTRIBUTING.md)           | Contribution guidelines                                   |
| [CLAUDE.md](CLAUDE.md)                       | Instructions for Claude Code                              |
