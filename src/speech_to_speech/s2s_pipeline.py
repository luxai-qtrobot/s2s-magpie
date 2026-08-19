from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from queue import Queue
from sys import platform
from threading import Event
from typing import Any

import nltk
import torch
from luxai.magpie.utils import Logger

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.backend_registry import (
    LLM_BACKENDS,
    STT_BACKENDS,
    TTS_BACKENDS,
    BackendSelection,
    BackendSpec,
    HandlerContext,
    create_backend_handler,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.queue_types import (
    AudioInItem,
    AudioOutItem,
    LMOutItem,
    STTOutItem,
    TextEventItem,
    TextPromptItem,
    TTSInItem,
    VADOutItem,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.VAD.vad_handler import VADHandler

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")

logging.getLogger("numba").setLevel(logging.WARNING)  # quiet down numba logs

@dataclass
class ParsedArguments:
    module_kwargs: ModuleArguments
    vad_handler_kwargs: VADHandlerArguments
    stt_backend: BackendSelection
    llm_backend: BackendSelection
    tts_backend: BackendSelection


_MISSING = object()


def _parameter_value(namespace: Any, name: str) -> Any:
    """Read one Paramify namespace value without coupling to Paramify internals."""

    if isinstance(namespace, Mapping):
        return namespace.get(name, _MISSING)
    return getattr(namespace, name, _MISSING)


def _dataclass_from_parameters(
    config_type: type[Any],
    namespace: Any,
    *,
    prefix: str | None = None,
) -> Any:
    """Overlay a Paramify group on a backend argument dataclass.

    Backend dataclasses retain their upstream defaults.  The Paramify schema
    uses concise names (for example ``tts.model_name``), while the vendored
    backend may use a registry prefix (``qwen3_tts_model_name``).  Both forms
    are accepted, with the concise form taking precedence.
    """

    config = config_type()
    marker = f"{prefix}_" if prefix else None
    for config_field in fields(config):
        candidates: list[str] = []
        if marker and config_field.name.startswith(marker):
            concise_name = config_field.name[len(marker) :]
            # ``tts.backend`` selects the registered backend (qwen3), while
            # ``tts.backend_engine`` selects Qwen's torch/ggml implementation.
            # Keep both concepts explicit instead of overloading one value.
            if concise_name == "backend":
                candidates.append("backend_engine")
            else:
                candidates.append(concise_name)
        candidates.append(config_field.name)

        for candidate in candidates:
            value = _parameter_value(namespace, candidate)
            if value is not _MISSING:
                setattr(config, config_field.name, value)
                break
    return config


def _select_backend(
    registry: Mapping[str, BackendSpec],
    name: str,
    namespace: Any,
) -> BackendSelection:
    try:
        spec = registry[name]
    except KeyError as exc:
        choices = ", ".join(registry)
        raise ValueError(f"Unsupported backend {name!r}; choose one of: {choices}.") from exc

    config = _dataclass_from_parameters(
        spec.config_type,
        namespace,
        prefix=spec.config_prefix,
    )
    return BackendSelection(spec, spec.normalize(config))


def build_arguments_from_parameters(parameters: Any) -> ParsedArguments:
    """Adapt the single Paramify configuration into the S2S core arguments.

    Paramify owns configuration-file loading and CLI overrides.  This adapter
    only selects registered backends and maps the relevant groups onto their
    existing dataclass defaults; it does not parse command-line arguments.
    """

    stt_name = str(parameters.stt.backend)
    llm_name = str(parameters.llm.backend)
    tts_name = str(parameters.tts.backend)

    module_kwargs = _dataclass_from_parameters(
        ModuleArguments,
        parameters.session,
    )
    module_kwargs.stt = stt_name
    module_kwargs.llm_backend = llm_name
    module_kwargs.tts = tts_name
    module_kwargs.log_level = str(parameters.log_level)

    args = ParsedArguments(
        module_kwargs=module_kwargs,
        vad_handler_kwargs=_dataclass_from_parameters(
            VADHandlerArguments,
            parameters.vad,
        ),
        stt_backend=_select_backend(STT_BACKENDS, stt_name, parameters.stt),
        llm_backend=_select_backend(LLM_BACKENDS, llm_name, parameters.llm),
        tts_backend=_select_backend(TTS_BACKENDS, tts_name, parameters.tts),
    )
    Logger.info(
        f"S2S backends configured: STT={stt_name}, LLM={llm_name}, TTS={tts_name}"
    )
    return args


def setup_logger(log_level: str) -> None:
    from speech_to_speech.pipeline.log_context import PipelineLogFilter

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(pipeline_prefix)s%(name)s - %(levelname)s - %(message)s",
    )
    # Attach the filter to every existing handler so each LogRecord gets a
    # `pipeline_prefix` attribute (matching the format string above).
    pipeline_filter = PipelineLogFilter()
    for h in logging.getLogger().handlers:
        h.addFilter(pipeline_filter)

    # torch compile logs
    if log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)


def check_mac_settings(module_kwargs: ModuleArguments) -> None:
    if platform == "darwin":
        if module_kwargs.device == "cuda":
            raise ValueError("Cannot use CUDA on macOS. Please set the device to 'cpu' or 'mps'.")
        if module_kwargs.llm_backend != "mlx-lm":
            Logger.warning(
                "For macOS users, it is recommended to use mlx-lm. "
                "You can activate it by passing --llm_backend mlx-lm."
            )
        if module_kwargs.tts not in ("pocket", "kokoro", "qwen3"):
            Logger.warning(
                "For macOS users, it is recommended to use qwen3 for TTS (pocket and kokoro are also valid options)."
            )


def prepare_module_args(module_kwargs: ModuleArguments, llm_backend: BackendSelection) -> None:
    if module_kwargs.tts is None:
        module_kwargs.tts = "qwen3"
    if module_kwargs.stt == "none" and not llm_backend.spec.capabilities.supports_audio_input:
        supported = ", ".join(name for name, spec in LLM_BACKENDS.items() if spec.capabilities.supports_audio_input)
        raise ValueError(f"--stt none requires an audio-input LLM backend; choose one of: {supported}.")
    if platform == "darwin":
        check_mac_settings(module_kwargs)


def prepare_all_args(args: ParsedArguments) -> None:
    """Validate selectors and apply the global device to selected configs only."""

    prepare_module_args(args.module_kwargs, args.llm_backend)
    if args.module_kwargs.device is None:
        return
    for field_name in ("stt_backend", "llm_backend", "tts_backend"):
        selection = getattr(args, field_name)
        if "device" not in selection.config:
            continue
        config = {**selection.config, "device": args.module_kwargs.device}
        setattr(args, field_name, replace(selection, config=config))


def _build_handlers(
    *,
    stop_event: Event,
    should_listen: Event,
    recv_audio_chunks_queue: Queue[AudioInItem],
    spoken_prompt_queue: Queue[VADOutItem],
    stt_output_queue: Queue[STTOutItem],
    text_prompt_queue: Queue[TextPromptItem],
    lm_response_queue: Queue[LMOutItem],
    lm_processed_queue: Queue[TTSInItem],
    send_audio_chunks_queue: Queue[AudioOutItem],
    text_output_queue: Queue[TextEventItem],
    module_kwargs: ModuleArguments,
    vad_handler_kwargs: VADHandlerArguments,
    stt_backend: BackendSelection,
    llm_backend: BackendSelection,
    tts_backend: BackendSelection,
    speculative_turns: SpeculativeTurnTracker,
    cancel_scope: CancelScope,
    pipeline_index: int,
) -> list[Any]:
    """Build a handler chain: VAD â†’ STT/AudioInput â†’ LM â†’ TTS."""
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs={
            **{
                config_field.name: deepcopy(getattr(vad_handler_kwargs, config_field.name))
                for config_field in fields(vad_handler_kwargs)
            },
            "text_output_queue": text_output_queue,
            "speculative_turns": speculative_turns,
        },
    )

    needs_notifier = not stt_backend.spec.capabilities.bypasses_transcription_notifier
    stt_queue_out: Queue[Any] = stt_output_queue if needs_notifier else text_prompt_queue
    stt_context = HandlerContext(
        stop_event=stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=stt_queue_out,
        text_output_queue=text_output_queue,
        should_listen=should_listen,
        cancel_scope=cancel_scope,
        speculative_turns=speculative_turns,
        pipeline_index=pipeline_index,
        sample_rate=vad_handler_kwargs.sample_rate,
        enable_live_transcription=module_kwargs.enable_live_transcription,
        live_transcription_update_interval=module_kwargs.live_transcription_update_interval,
    )
    speech_input_handlers = [create_backend_handler(stt_backend, stt_context)]
    if needs_notifier:
        transcription_notifier = TranscriptionNotifier(
            stop_event,
            queue_in=stt_output_queue,
            queue_out=text_prompt_queue,  # type: ignore[arg-type]
            setup_kwargs={
                "text_output_queue": text_output_queue,
                "should_listen": should_listen,
            },
        )
        speech_input_handlers.append(transcription_notifier)

    def handler_context(queue_in: Queue[Any], queue_out: Queue[Any]) -> HandlerContext:
        return HandlerContext(
            stop_event=stop_event,
            queue_in=queue_in,
            queue_out=queue_out,
            text_output_queue=text_output_queue,
            should_listen=should_listen,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
            pipeline_index=pipeline_index,
            sample_rate=vad_handler_kwargs.sample_rate,
            enable_live_transcription=module_kwargs.enable_live_transcription,
            live_transcription_update_interval=module_kwargs.live_transcription_update_interval,
        )

    lm_context = handler_context(text_prompt_queue, lm_response_queue)
    lm = create_backend_handler(
        llm_backend,
        lm_context,
    )

    lm_processor = LMOutputProcessor(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=lm_processed_queue,
        setup_kwargs={
            "speculative_turns": speculative_turns,
            "text_output_queue": text_output_queue,
        },
    )

    tts_context = handler_context(lm_processed_queue, send_audio_chunks_queue)
    tts = create_backend_handler(
        tts_backend,
        tts_context,
    )

    return [vad, *speech_input_handlers, lm, lm_processor, tts]


def build_pipeline_unit(
    *,
    index: int,
    stop_event: Event,
    module_kwargs: ModuleArguments,
    vad_handler_kwargs: VADHandlerArguments,
    stt_backend: BackendSelection,
    llm_backend: BackendSelection,
    tts_backend: BackendSelection,
) -> "PipelineUnit":
    """Build one isolated pipeline with its own state and queues.

    Handler instances are returned in ``unit.handlers`` so the caller can run
    them with ``ThreadManager``. Building a unit does not create a network
    server or select a session transport.
    """
    from speech_to_speech.api.openai_realtime.service import RealtimeService

    # Per-unit copies isolate any setup-time mutation performed by third-party libraries.
    vad_kw = deepcopy(vad_handler_kwargs)
    stt_selection = stt_backend.copy_for_pipeline()
    llm_selection = llm_backend.copy_for_pipeline()
    tts_selection = tts_backend.copy_for_pipeline()

    should_listen = Event()
    response_playing = Event()
    cancel_scope = CancelScope()
    speculative_turns = SpeculativeTurnTracker()
    recv_audio_chunks_queue: Queue[AudioInItem] = Queue()
    send_audio_chunks_queue: Queue[AudioOutItem] = Queue()
    spoken_prompt_queue: Queue[VADOutItem] = Queue()
    stt_output_queue: Queue[STTOutItem] = Queue()
    text_prompt_queue: Queue[TextPromptItem] = Queue()
    lm_response_queue: Queue[LMOutItem] = Queue()
    lm_processed_queue: Queue[TTSInItem] = Queue()
    text_output_queue: Queue[TextEventItem] = Queue()

    chat_size = llm_selection.config.get("chat_size", 10)
    default_instructions = llm_selection.config.get("init_chat_prompt")

    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
        chat_size=chat_size,
        speculative_turns=speculative_turns,
        default_instructions=default_instructions,
    )

    if module_kwargs.enable_live_transcription:
        vad_kw.enable_realtime_transcription = True
        vad_kw.realtime_processing_pause = module_kwargs.live_transcription_update_interval

    handlers = _build_handlers(
        stop_event=stop_event,
        should_listen=should_listen,
        recv_audio_chunks_queue=recv_audio_chunks_queue,
        spoken_prompt_queue=spoken_prompt_queue,
        stt_output_queue=stt_output_queue,
        text_prompt_queue=text_prompt_queue,
        lm_response_queue=lm_response_queue,
        lm_processed_queue=lm_processed_queue,
        send_audio_chunks_queue=send_audio_chunks_queue,
        text_output_queue=text_output_queue,
        module_kwargs=module_kwargs,
        vad_handler_kwargs=vad_kw,
        stt_backend=stt_selection,
        llm_backend=llm_selection,
        tts_backend=tts_selection,
        speculative_turns=speculative_turns,
        cancel_scope=cancel_scope,
        pipeline_index=index,
    )
    for h in handlers:
        h.pipeline_index = index

    return PipelineUnit(
        index=index,
        service=service,
        cancel_scope=cancel_scope,
        should_listen=should_listen,
        response_playing=response_playing,
        input_queue=recv_audio_chunks_queue,
        output_queue=send_audio_chunks_queue,
        text_output_queue=text_output_queue,
        text_prompt_queue=text_prompt_queue,
        handlers=handlers,
    )
