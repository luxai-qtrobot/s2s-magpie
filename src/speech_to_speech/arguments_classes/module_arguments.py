from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.backend_registry import LLM_BACKENDS, STT_BACKENDS, TTS_BACKENDS

_AUDIO_INPUT_LLM_BACKENDS = ", ".join(
    name for name, spec in LLM_BACKENDS.items() if spec.capabilities.supports_audio_input
)


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mac_optimal_settings: bool = field(
        default=False,
        metadata={
            "help": "If specified, provides macOS defaults: Parakeet TDT for STT, MLX LM for the language "
            "model, Qwen3-TTS for TTS, and MPS for supported component devices. Explicit component, model, "
            "global-device, and component-device flags override these defaults. It does not select a command.",
        },
    )
    stt: Optional[str] = field(
        default="parakeet-tdt",
        metadata={
            "choices": tuple(STT_BACKENDS),
            "help": "The STT to use. Use 'none' to send VAD audio directly to an audio-input LLM. "
            f"Audio-input LLM backends: {_AUDIO_INPUT_LLM_BACKENDS}. Select an explicitly audio-capable "
            "model with --model_name. Default is 'parakeet-tdt'.",
        },
    )
    llm_backend: Optional[str] = field(
        default="responses-api",
        metadata={
            "choices": tuple(LLM_BACKENDS),
            "help": "The LLM backend to use. Default is 'responses-api'.",
        },
    )
    tts: Optional[str] = field(
        default="qwen3",
        metadata={
            "choices": tuple(TTS_BACKENDS),
            "help": "The TTS backend to use. Default is 'qwen3'.",
        },
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Provide logging level. Example --log_level debug, default=info."},
    )
    enable_live_transcription: bool = field(
        default=True,
        metadata={
            "help": "Enable live transcription display while user is speaking (works with parakeet-tdt). Default is true."
        },
    )
    live_transcription_update_interval: float = field(
        default=0.5,
        metadata={"help": "Update interval for live transcription in seconds (default: 0.5s = 500ms)"},
    )
    live_transcription_min_silence_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum silence duration (ms) before ending speech when live transcription is enabled (default: 500ms)"
        },
    )
