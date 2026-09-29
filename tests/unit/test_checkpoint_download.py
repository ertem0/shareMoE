import json
from pathlib import Path

import pytest
from huggingface_hub.errors import HTTPError

from moe_engine.checkpoint import download
from moe_engine.checkpoint.download import (
    MARKER_NAME,
    CheckpointDownloadError,
    CheckpointMismatchError,
    CheckpointSpec,
    InvalidCheckpointSpecError,
    ensure_checkpoint,
    hf_download,
)

REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"


class RecordingDownloader:
    """Writes a placeholder file instead of contacting the Hub."""

    def __init__(self) -> None:
        self.calls: list[CheckpointSpec] = []

    def __call__(self, spec: CheckpointSpec) -> None:
        self.calls.append(spec)
        (spec.local_dir / "config.json").write_text("{}")


def failing_downloader(spec: CheckpointSpec) -> None:
    raise CheckpointDownloadError("offline")


def make_spec(local_dir: Path, revision: str = REVISION) -> CheckpointSpec:
    return CheckpointSpec(
        model_id="owner/model", revision=revision, local_dir=local_dir
    )


def test_downloads_into_the_configured_directory(tmp_path: Path) -> None:
    spec = make_spec(tmp_path / "nested" / "checkpoint")
    downloader = RecordingDownloader()

    result = ensure_checkpoint(spec, downloader)

    assert result == spec.local_dir
    assert downloader.calls == [spec]
    assert (spec.local_dir / "config.json").exists()
    assert json.loads((spec.local_dir / MARKER_NAME).read_text()) == spec.marker()


def test_later_start_reuses_the_checkpoint_without_downloading(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    downloader = RecordingDownloader()
    ensure_checkpoint(spec, downloader)

    result = ensure_checkpoint(spec, downloader)

    assert result == spec.local_dir
    assert len(downloader.calls) == 1


def test_reuse_works_when_downloads_are_impossible(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    ensure_checkpoint(spec, RecordingDownloader())

    assert ensure_checkpoint(spec, failing_downloader) == spec.local_dir


def test_failed_download_leaves_no_marker(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)

    with pytest.raises(CheckpointDownloadError):
        ensure_checkpoint(spec, failing_downloader)

    assert not (tmp_path / MARKER_NAME).exists()


def test_directory_with_a_different_revision_is_rejected(tmp_path: Path) -> None:
    ensure_checkpoint(make_spec(tmp_path), RecordingDownloader())
    other = make_spec(tmp_path, revision="0" * 40)

    with pytest.raises(CheckpointMismatchError):
        ensure_checkpoint(other, RecordingDownloader())


def test_unreadable_marker_is_rejected(tmp_path: Path) -> None:
    (tmp_path / MARKER_NAME).write_text("not json")

    with pytest.raises(CheckpointMismatchError):
        ensure_checkpoint(make_spec(tmp_path), RecordingDownloader())


@pytest.mark.parametrize("revision", ["main", REVISION[:7], REVISION.upper(), ""])
def test_spec_requires_a_pinned_commit(tmp_path: Path, revision: str) -> None:
    with pytest.raises(InvalidCheckpointSpecError):
        make_spec(tmp_path, revision=revision)


@pytest.mark.parametrize("model_id", ["", "model", "a/b/c"])
def test_spec_rejects_malformed_model_ids(tmp_path: Path, model_id: str) -> None:
    with pytest.raises(InvalidCheckpointSpecError):
        CheckpointSpec(model_id=model_id, revision=REVISION, local_dir=tmp_path)


def test_hf_download_requests_the_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        download,
        "snapshot_download",
        lambda repo_id, **kwargs: calls.append((repo_id, kwargs)),
    )
    spec = make_spec(tmp_path)

    hf_download(spec)

    assert calls == [
        (
            "owner/model",
            {
                "revision": REVISION,
                "local_dir": tmp_path,
                "allow_patterns": ["*.json", "*.safetensors"],
            },
        )
    ]


@pytest.mark.parametrize(
    "error", [HTTPError("connection refused"), OSError("disk full")]
)
def test_hf_download_failures_become_controlled_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(repo_id: str, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(download, "snapshot_download", fail)

    with pytest.raises(CheckpointDownloadError) as caught:
        hf_download(make_spec(tmp_path))

    assert caught.value.__cause__ is error
