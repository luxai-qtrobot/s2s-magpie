from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from speech_to_speech.model_assets import (
    load_asset_lock,
    locked_revision_for_repo,
    resolve_huggingface_file,
)


def test_production_assets_have_immutable_revisions() -> None:
    lock = load_asset_lock()
    revisions = {asset["revision"] for asset in lock["huggingface"].values()}

    assert all(len(revision) == 40 for revision in revisions)
    assert set(lock["nltk"]) == {"punkt_tab"}


def test_optional_qwen_base_model_has_a_locked_revision() -> None:
    lock = load_asset_lock()
    asset = lock["huggingface"]["qwen3_tts_0_6b_base"]

    assert asset["optional"] is True
    assert locked_revision_for_repo(asset["repo_id"]) == asset["revision"]


def test_huggingface_resolution_is_pinned_and_offline(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.nemo"
    checkpoint.write_bytes(b"test")
    calls = []
    fake_hub = ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    resolved = resolve_huggingface_file(
        "nvidia/parakeet-tdt-0.6b-v3",
        filename="parakeet-tdt-0.6b-v3.nemo",
        revision=None,
    )

    assert resolved == checkpoint.resolve()
    assert calls[0]["revision"] == "b51b7dc0fbf7f266a97880fb4b626c56d28f4b96"
    assert calls[0]["local_files_only"] is True


def test_startup_sources_do_not_download_mutable_assets() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "speech_to_speech"
    pipeline_source = (root / "s2s_pipeline.py").read_text(encoding="utf-8")
    vad_source = (root / "VAD" / "vad_handler.py").read_text(encoding="utf-8")

    assert "nltk.download" not in pipeline_source
    assert "torch.hub.load" not in vad_source
    assert "silero-vad:master" not in vad_source
