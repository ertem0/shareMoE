# Distributed MoE inference engine

Research prototype that distributes the experts of an MoE model across worker nodes. One coordinator hosts every non-expert component and dispatches hidden states to the workers that hold the routed experts. Prioritize correctness, modularity and reproducibility over optimization.

## Before starting a task

- Read docs/PROJECT.md and docs/ARCHITECTURE.md.
- Work only on the milestone marked `in progress` in docs/PROJECT.md. Ask me if the task doesn't clearly belong to it.
- Never implement anything under Future Scope unless the task says so.
- Inspect existing code before adding abstractions. Follow the project structure in docs/ARCHITECTURE.md and ask before adding new top-level modules.

## Commands

Always run through `uv run`, never an activated venv or pip.

- Setup: `uv sync`
- Tests: `uv run pytest` (unit in tests/unit, integration in tests/integration)
- Before committing: `uv run ruff format . && uv run ruff check . && uv run pytest`

## Architecture rules

- Model logic never imports transport code. The dense runtime calls the expert executor interface.
- OLMoE-specific code lives only in `models/olmoe.py`. Everything else is model-agnostic.
- `ExpertWorker` does no I/O. No networking, downloads or checkpoint access inside it.
- The coordinator's expert directory is the only source of truth for expert ownership.
- Bulk data (weights, hidden states) goes over the tensor connection. The control connection carries only small metadata messages.
- If a change affects the wire protocol, message types or a sequence, update docs/ARCHITECTURE.md in the same change.

## Testing rules

- Every new behavior needs a test.
- Tests are offline. Use small local fixtures, never Hugging Face downloads.
- Compare tensors with `torch.testing.assert_close` and dtype-appropriate tolerances, never exact equality.
- Multi-process tests use explicit synchronization and deterministic request IDs, not sleeps or timing.

## Error handling

- Fail explicitly with controlled errors on unknown expert or layer IDs, malformed frames, incompatible tensor metadata, unavailable workers, connection failures and timeouts.
- No broad `except Exception` or bare `except`.

## After finishing a milestone

Update its status in docs/PROJECT.md and tell me which acceptance criteria are covered by which tests.

## Git

### Commit messages

Use Conventional Commits: `<type>(<scope>): <summary>`

- Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `perf`
- Scopes: `experts`, `models`, `checkpoint`, `protocol`, `transport`, `coordinator`, `worker`, `bench`, `docs`, `deps`
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- Add a body when the reason for the change isn't obvious. Wrap it at 72 characters.
- Reference the milestone in the body, e.g. `Milestone: 4`.

Examples:

- `feat(protocol): add frame encoder and decoder`
- `test(coordinator): cover duplicate expert reservation`
- `fix(transport): fail explicitly on truncated frames`

### Workflow

- One logical change per commit. Don't mix refactors with features.
- Commit directly to `main` in small commits. Use a branch only when I ask for an experiment.
- Run format, lint and tests before every commit. Don't commit if any of them fail.
- Never amend or rewrite commits that were already pushed.
- Never push. I push after reviewing.
