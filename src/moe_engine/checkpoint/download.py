"""Downloads a pinned checkpoint once and reuses it on later starts.

This module only places files on local storage. It knows nothing about the
model architecture or the weights inside the files.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, HTTPError

# Written after a download completes. Its presence means the directory holds
# the complete checkpoint it describes.
MARKER_NAME = ".moe_engine_checkpoint.json"

DEFAULT_ALLOW_PATTERNS = ("*.json", "*.safetensors")

_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


class InvalidCheckpointSpecError(ValueError):
    """A checkpoint specification is malformed or not pinned."""


class CheckpointDownloadError(RuntimeError):
    """Downloading the checkpoint failed."""


class CheckpointMismatchError(RuntimeError):
    """The local directory holds a different or unreadable checkpoint."""


@dataclass(frozen=True)
class CheckpointSpec:
    """Which checkpoint to use and where to keep it."""

    model_id: str
    revision: str
    local_dir: Path
    allow_patterns: tuple[str, ...] = DEFAULT_ALLOW_PATTERNS

    def __post_init__(self) -> None:
        if not self.model_id or self.model_id.count("/") != 1:
            raise InvalidCheckpointSpecError(
                f"model ID must look like 'owner/name', got {self.model_id!r}"
            )
        if not _COMMIT_HASH.fullmatch(self.revision):
            raise InvalidCheckpointSpecError(
                "revision must be a full 40-character commit hash so the "
                f"checkpoint is pinned, got {self.revision!r}"
            )
        if not self.allow_patterns:
            raise InvalidCheckpointSpecError("allow_patterns must not be empty")

    def marker(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "allow_patterns": list(self.allow_patterns),
        }


Downloader = Callable[[CheckpointSpec], None]


def hf_download(spec: CheckpointSpec) -> None:
    """Download the checkpoint files from the Hugging Face Hub."""
    try:
        snapshot_download(
            spec.model_id,
            revision=spec.revision,
            local_dir=spec.local_dir,
            allow_patterns=list(spec.allow_patterns),
        )
    except (HTTPError, HFValidationError, OSError) as exc:
        raise CheckpointDownloadError(
            f"could not download {spec.model_id}@{spec.revision}: {exc}"
        ) from exc


def ensure_checkpoint(
    spec: CheckpointSpec, downloader: Downloader = hf_download
) -> Path:
    """Return the checkpoint directory, downloading it only if needed.

    A directory whose marker matches `spec` is reused without contacting the
    Hub, so this works offline once the checkpoint is present.
    """
    marker_path = spec.local_dir / MARKER_NAME
    if marker_path.exists():
        recorded = _read_marker(marker_path)
        if recorded != spec.marker():
            raise CheckpointMismatchError(
                f"{spec.local_dir} holds {recorded}, expected {spec.marker()}"
            )
        return spec.local_dir

    spec.local_dir.mkdir(parents=True, exist_ok=True)
    downloader(spec)
    _write_marker(marker_path, spec)
    return spec.local_dir


def _read_marker(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointMismatchError(f"unreadable checkpoint marker {path}") from exc


def _write_marker(path: Path, spec: CheckpointSpec) -> None:
    # Write then rename, so an interrupted write never leaves a marker behind.
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(json.dumps(spec.marker(), indent=2))
    partial.replace(path)
