from __future__ import annotations

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")

VADIterator = import_module("speech_to_speech.VAD.vad_iterator").VADIterator


class _SpeechModel:
    def reset_states(self) -> None:
        pass

    def __call__(self, _audio, _sampling_rate):
        return torch.tensor(1.0)


def test_continuous_speech_is_returned_at_configured_limit() -> None:
    iterator = VADIterator(
        _SpeechModel(),
        threshold=0.5,
        sampling_rate=8000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
        max_speech_duration_ms=100,
    )
    chunk = torch.zeros(400)  # 50 ms at 8 kHz

    assert iterator(chunk) is None
    utterance = iterator(chunk)

    assert utterance is not None
    assert sum(len(part) for part in utterance) == 800
    assert iterator.last_utterance_forced is True
    assert iterator.triggered is False
    assert iterator.speech_buffer() == []

    # Continued speech starts another independently bounded segment.
    assert iterator(chunk) is None
    assert iterator.triggered is True
