import math
from collections import deque

import torch


class VADIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
        max_speech_duration_ms: float | None = 30_000.0,
    ) -> None:
        """
        Mainly taken from https://github.com/snakers4/silero-vad
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 300 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Retain up to speech_pad_ms of audio before VAD triggers and prepend it
            to the detected speech chunk

        max_speech_duration_ms: float | None (default - 30000 milliseconds)
            Bound retained speech audio. Reaching the limit returns the current
            utterance with ``last_utterance_forced`` set so the caller can
            create a safe segment boundary. ``None`` or infinity disables it.
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.is_speaking = False
        self.buffer: list[torch.Tensor] = []
        self.prefix_buffer: list[torch.Tensor] = []
        self.active_speech_samples = 0
        self.last_utterance_active_speech_samples = 0
        self.last_utterance_forced = False
        self._buffered_speech_samples = 0
        self._pre_speech_buffer: deque[torch.Tensor] = deque()
        self._pre_speech_samples = 0

        if sampling_rate not in [8000, 16000]:
            raise ValueError("VADIterator does not support sampling rates other than [8000, 16000]")

        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
        if max_speech_duration_ms is None or max_speech_duration_ms == math.inf:
            self.max_speech_samples: int | None = None
        else:
            if not math.isfinite(max_speech_duration_ms) or max_speech_duration_ms <= 0:
                raise ValueError("max_speech_duration_ms must be a positive finite value, infinity, or None")
            self.max_speech_samples = max(1, int(sampling_rate * max_speech_duration_ms / 1000))
        self.reset_states()

    def reset_states(self) -> None:
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.buffer = []
        self.prefix_buffer = []
        self.active_speech_samples = 0
        self.last_utterance_active_speech_samples = 0
        self.last_utterance_forced = False
        self._buffered_speech_samples = 0
        self._pre_speech_buffer.clear()
        self._pre_speech_samples = 0

    def _num_samples(self, chunk: torch.Tensor) -> int:
        return len(chunk[0]) if chunk.dim() == 2 else len(chunk)

    def _trim_pre_speech_buffer(self) -> None:
        while (
            self.speech_pad_samples > 0
            and self._pre_speech_buffer
            and self._pre_speech_samples > self.speech_pad_samples
        ):
            first = self._pre_speech_buffer[0]
            first_samples = self._num_samples(first)
            excess = self._pre_speech_samples - self.speech_pad_samples

            if excess >= first_samples:
                self._pre_speech_buffer.popleft()
                self._pre_speech_samples -= first_samples
                continue

            if first.dim() == 2:
                self._pre_speech_buffer[0] = first[:, excess:]
            else:
                self._pre_speech_buffer[0] = first[excess:]
            self._pre_speech_samples -= excess

    def _remember_pre_speech(self, chunk: torch.Tensor) -> None:
        if self.speech_pad_samples <= 0:
            self._pre_speech_buffer.clear()
            self._pre_speech_samples = 0
            return

        self._pre_speech_buffer.append(chunk)
        self._pre_speech_samples += self._num_samples(chunk)
        self._trim_pre_speech_buffer()

    def _speech_buffer(self) -> list[torch.Tensor]:
        if not self.prefix_buffer:
            return list(self.buffer)
        return [*self.prefix_buffer, *self.buffer]

    def speech_buffer(self) -> list[torch.Tensor]:
        return self._speech_buffer()

    def _finish_utterance(self, *, forced: bool) -> list[torch.Tensor]:
        spoken_utterance = self.speech_buffer()
        self.last_utterance_active_speech_samples = self.active_speech_samples
        self.last_utterance_forced = forced
        self.temp_end = 0
        self.triggered = False
        self.active_speech_samples = 0
        self._buffered_speech_samples = 0
        self.buffer = []
        self.prefix_buffer = []
        return spoken_utterance

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> list[torch.Tensor] | None:
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            self.prefix_buffer = list(self._pre_speech_buffer)
            self._pre_speech_buffer.clear()
            self._pre_speech_samples = 0
            self.buffer.append(x)
            self.active_speech_samples = window_size_samples
            self._buffered_speech_samples = sum(self._num_samples(chunk) for chunk in self.prefix_buffer)
            self._buffered_speech_samples += window_size_samples
            self.last_utterance_active_speech_samples = 0
            self.last_utterance_forced = False
            if self.max_speech_samples is not None and self._buffered_speech_samples >= self.max_speech_samples:
                return self._finish_utterance(forced=True)
            return None

        if not self.triggered:
            self._remember_pre_speech(x)
            return None

        if self.triggered:
            self.buffer.append(x)
            self._buffered_speech_samples += window_size_samples
            if speech_prob >= self.threshold - 0.15:
                self.active_speech_samples += window_size_samples
                if self.temp_end:
                    self.temp_end = 0

            if self.max_speech_samples is not None and self._buffered_speech_samples >= self.max_speech_samples:
                return self._finish_utterance(forced=True)

            if speech_prob < self.threshold - 0.15:
                if not self.temp_end:
                    self.temp_end = self.current_sample
                if self.current_sample - self.temp_end < self.min_silence_samples:
                    return None

                # End of speech: keep the final low-confidence chunks that were
                # observed before VAD decided the utterance was done.
                return self._finish_utterance(forced=False)

        return None
