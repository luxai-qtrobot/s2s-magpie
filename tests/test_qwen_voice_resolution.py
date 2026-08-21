from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


def _runtime_config(voice: str) -> SimpleNamespace:
    return SimpleNamespace(
        session=SimpleNamespace(
            audio=SimpleNamespace(output=SimpleNamespace(voice=voice)),
        )
    )


def _handler() -> Qwen3TTSHandler:
    handler = object.__new__(Qwen3TTSHandler)
    handler.ref_audio = "previous.wav"
    handler.ref_spk = Path("previous.spk")
    handler.ref_rvq = Path("previous.rvq")
    return handler


def test_bundled_voice_name_resolves_from_voices_directory(monkeypatch, tmp_path) -> None:
    voice = tmp_path / "voices" / "aiden.wav"
    voice.parent.mkdir()
    voice.write_bytes(b"wav")
    monkeypatch.chdir(tmp_path)
    handler = _handler()

    handler._apply_session_voice_override("base", _runtime_config("Aiden"))

    assert Path(handler.ref_audio) == voice.resolve()
    assert handler.ref_spk is None
    assert handler.ref_rvq is None


def test_absolute_voice_path_is_accepted(tmp_path) -> None:
    voice = tmp_path / "myvoice.wav"
    voice.write_bytes(b"wav")
    handler = _handler()

    handler._apply_session_voice_override("base", _runtime_config(str(voice)))

    assert Path(handler.ref_audio) == voice.resolve()


def test_unknown_voice_does_not_replace_the_current_reference(caplog) -> None:
    handler = _handler()

    handler._apply_session_voice_override("base", _runtime_config("missing"))

    assert handler.ref_audio == "previous.wav"
    assert handler.ref_spk == Path("previous.spk")
    assert handler.ref_rvq == Path("previous.rvq")
    assert "Rejecting unknown Qwen3-TTS voice" in caplog.text
