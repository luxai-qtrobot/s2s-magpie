import asyncio
from queue import Queue
from threading import Event
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.runtime.sink import SessionSink


class SessionState(BaseModel):
    """Per-client ephemeral state.

    Created when SessionRuntime attaches a sink to a PipelineUnit and dropped
    after the session drains. Holding the sink reference, service session id,
    and any send-loop scratch
    (pending_output_item) here ensures these fields share one lifecycle — a
    stale value can't outlive its session.

    ``sink`` may be None while a caller reserves or constructs a session,
    so output orchestration must tolerate a sink-less snapshot.

    `drained` is set by the send loop when SESSION_END travels through the handler
    chain back to the output queue; the release path awaits it before clearing
    `PipelineUnit.session`, so a new client cannot claim the unit until in-flight
    work from this session has fully reset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sink: Optional[SessionSink] = None
    session_id: str = ""
    pending_output_item: Any = None
    # Response-keyed side-channel events can be generated before a hidden
    # prefetch is claimed. Keep them ordered and private without letting one
    # blocked event stall unrelated origin-response output.
    pending_text_output_items: list[Any] = Field(default_factory=list)
    drained: asyncio.Event = Field(default_factory=asyncio.Event)
    # Wall-clock time when close began. None while the client is active.
    released_at: Optional[float] = None
    # Wall-clock time when the drain wait gave up and quarantined the unit
    # (SESSION_END_QUARANTINE_TIMEOUT_S elapsed). The unit stays unclaimable —
    # its handlers may still emit this session's output — until SESSION_END
    # actually drains.
    quarantined_at: Optional[float] = None


class PipelineUnit(BaseModel):
    """One isolated realtime pipeline.

    Each unit owns its queues, events, RealtimeService, and the chain of handler
    instances (VAD, STT, transcription notifier, LM, LM output processor, TTS).
    A SessionRuntime claims the unit while ``session`` is non-None and only
    releases it after SESSION_END has propagated through the full handler chain.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    service: RealtimeService
    cancel_scope: CancelScope
    should_listen: Event
    response_playing: Event
    input_queue: Queue
    output_queue: Queue
    text_output_queue: Queue
    text_prompt_queue: Queue
    handlers: list[Any]
    pipeline_queues: dict[str, Queue] = Field(default_factory=dict)

    session: Optional[SessionState] = None

    def queue_metrics(self) -> dict[str, dict[str, int | str]]:
        """Return bounded-queue runtime counters for health/diagnostics."""

        snapshots: dict[str, dict[str, int | str]] = {}
        for name, queue in self.pipeline_queues.items():
            metrics = getattr(queue, "metrics", None)
            if callable(metrics):
                snapshots[name] = metrics().as_dict()
            else:
                snapshots[name] = {
                    "name": name,
                    "size": queue.qsize(),
                    "max_size": queue.maxsize,
                    "high_watermark": queue.qsize(),
                    "overloads": 0,
                    "dropped_items": 0,
                }
        return snapshots
