"""Smart Turn v3.2 end-of-turn inference.

Smart Turn complements a conventional VAD by looking at the acoustic and
linguistic content of a complete utterance.  It is only invoked after Silero
has found a speech-to-silence boundary, so it does not add work to every audio
chunk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from speech_to_speech.model_assets import locked_huggingface_asset, resolve_huggingface_file

logger = logging.getLogger(__name__)

MODEL_VERSION = "v3.2"
MODEL_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 8
MODEL_FILENAME = f"smart-turn-{MODEL_VERSION}-cpu.onnx"


@dataclass(frozen=True)
class SmartTurnResult:
    """Result of one Smart Turn prediction."""

    complete: bool
    probability: float
    inference_ms: float


class SmartTurnAnalyzer:
    """Run the Smart Turn v3.2 ONNX model on up to eight seconds of audio."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        model_revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = True,
        threshold: float = 0.5,
        cpu_count: int = 1,
        warmup: bool = True,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Smart Turn threshold must be between 0 and 1, got {threshold}")
        if cpu_count < 1:
            raise ValueError(f"Smart Turn cpu_count must be at least 1, got {cpu_count}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("Smart Turn requires the `onnxruntime` package for CPU inference.") from exc

        from transformers import WhisperFeatureExtractor

        self.threshold = threshold
        locked_asset = locked_huggingface_asset("smart_turn_v3_2_cpu")
        model_reference = model_path or str(locked_asset["repo_id"])
        self.model_path = resolve_huggingface_file(
            model_reference,
            filename=MODEL_FILENAME,
            revision=model_revision or str(locked_asset["revision"]),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = cpu_count
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=MAX_AUDIO_SECONDS)

        logger.info(
            "Loaded Smart Turn %s from %s using %s (threshold=%.2f)",
            MODEL_VERSION,
            self.model_path,
            "CPUExecutionProvider",
            threshold,
        )
        if warmup:
            self.predict(np.zeros(MODEL_SAMPLE_RATE, dtype=np.float32))

    @staticmethod
    def _prepare_audio(audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"Smart Turn expects mono 1-D audio, got shape {audio.shape}")
        if sample_rate <= 0:
            raise ValueError(f"Smart Turn sample rate must be positive, got {sample_rate}")

        if sample_rate != MODEL_SAMPLE_RATE:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(sample_rate, MODEL_SAMPLE_RATE)
            audio = resample_poly(
                audio,
                MODEL_SAMPLE_RATE // divisor,
                sample_rate // divisor,
            ).astype(np.float32, copy=False)

        max_samples = MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        elif audio.size < max_samples:
            audio = np.pad(audio, (max_samples - audio.size, 0), mode="constant")
        return audio

    def predict(self, audio_array: np.ndarray, *, sample_rate: int = MODEL_SAMPLE_RATE) -> SmartTurnResult:
        """Return whether ``audio_array`` sounds like a completed user turn."""
        started_at = time.perf_counter()
        audio = self._prepare_audio(audio_array, sample_rate)
        inputs = self.feature_extractor(
            audio,
            sampling_rate=MODEL_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )
        input_features = np.asarray(inputs.input_features, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: input_features})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not np.isfinite(probability):
            raise RuntimeError(f"Smart Turn returned a non-finite probability: {probability}")

        return SmartTurnResult(
            complete=probability > self.threshold,
            probability=probability,
            inference_ms=(time.perf_counter() - started_at) * 1000,
        )
