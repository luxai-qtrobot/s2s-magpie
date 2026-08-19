"""Transport-independent lifecycle and output orchestration for one session.

This module contains the session logic that originally lived inside the
FastAPI realtime router.  It deliberately knows nothing about HTTP,
WebSockets, WebRTC, or MAGPIE.  A :class:`SessionSink` receives typed realtime
events and raw pipeline-rate PCM.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, TypeVar

import numpy as np
from luxai.magpie.utils import Logger
from openai.types.realtime import (
    ConversationItemCreateEvent,
    ConversationItemTruncateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    OutputAudioBufferClearEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    SessionUpdateEvent,
)

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit, SessionState
from speech_to_speech.api.openai_realtime.service import PIPELINE_SAMPLE_RATE
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    AssistantToolCallReadyEvent,
    AudioInputCompletedEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput
from speech_to_speech.runtime.sink import SessionSink

MAX_AUDIO_BATCH_BYTES = 6400
SESSION_END_DRAIN_TIMEOUT_S = 10.0
SESSION_END_QUARANTINE_TIMEOUT_S = 180.0

QItem = TypeVar("QItem")


def _keep_cancel_bookkeeping(item: Any) -> bool:
    """Keep accounting and lifecycle terminals while cancelling output."""

    return isinstance(item, TokenUsageEvent) or _is_audio_done(item) or is_control_message(item, SESSION_END.kind)


def _keep_user_text_event(item: Any) -> bool:
    return isinstance(
        item,
        (
            SpeechStoppedEvent,
            PartialTranscriptionEvent,
            TranscriptionCompletedEvent,
            AudioInputCompletedEvent,
        ),
    )


def _keep_pipeline_control(item: Any) -> bool:
    return isinstance(item, (PipelineControlMessage, bytes))


def _audio_payload(item: Any) -> Any:
    return item.audio if isinstance(item, AudioOutput) else item


def _audio_generation(item: Any) -> int | None:
    return item.cancel_generation if isinstance(item, AudioOutput) else None


def _audio_response_key(item: Any) -> str | None:
    return item.response_key if isinstance(item, AudioOutput) else None


def _audio_cleanup_only(item: Any) -> bool:
    return item.cleanup_only if isinstance(item, AudioOutput) else False


_RESPONSE_PIPELINE_EVENTS = (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    ResponseFailedEvent,
)


def _keep_non_audio_output(item: Any) -> bool:
    """Preserve response bookkeeping when buffered audio is cleared."""

    return _keep_cancel_bookkeeping(item) or isinstance(item, _RESPONSE_PIPELINE_EVENTS)


def _response_event_key(item: Any) -> str | None:
    if isinstance(item, _RESPONSE_PIPELINE_EVENTS):
        return item.response_key
    return None


def _output_response_key(item: Any) -> str | None:
    if isinstance(item, AudioOutput):
        return item.response_key
    if isinstance(item, PipelineEvent):
        return getattr(item, "response_key", None)
    return None


def _response_key_is_obsolete(unit: PipelineUnit, session_id: str, response_key: str | None) -> bool:
    """Whether *response_key* belongs to a closed response rather than a queued one."""

    if response_key is None:
        return False
    state = unit.service._state(session_id)
    if response_key in state.closed_response_keys:
        return True
    return (
        state.in_response
        and state.current_response_key not in (None, response_key)
        and response_key not in state.pending_response_keys
    )


def _response_key_output_is_blocked(
    unit: PipelineUnit,
    session_id: str,
    response_key: str | None,
) -> bool:
    if response_key is None:
        return False
    return unit.service.response.is_response_output_blocked(session_id, response_key)


def _discard_obsolete_response_key(unit: PipelineUnit, session_id: str, response_key: str | None) -> None:
    if response_key is None:
        return
    unit.service.close_response_key(session_id, response_key)
    Logger.debug(f"Pipeline {unit.index}: discarded obsolete response {response_key} output")


def _flush_queue(
    queue: Queue[QItem],
    *,
    preserve: Callable[[QItem], bool] | None = None,
    on_discard: Callable[[QItem], None] | None = None,
) -> None:
    """Drain a queue, optionally preserving matching items at its front."""

    preserved: list[QItem] = []
    while True:
        try:
            item = queue.get_nowait()
            if preserve and preserve(item):
                preserved.append(item)
            elif on_discard is not None:
                on_discard(item)
        except Empty:
            break
    if preserved:
        # Queue has no public prepend operation. Holding its mutex makes this
        # atomic with concurrent producers and preserves terminal ordering.
        with queue.mutex:
            for item in reversed(preserved):
                queue.queue.appendleft(item)
            queue.not_empty.notify(len(preserved))


def _clean_unit(
    unit: PipelineUnit,
    preserve: Callable[[Any], bool] | None = None,
    on_discard: Callable[[Any], None] | None = None,
) -> None:
    """Cancel in-flight work and clear every externally visible edge queue."""

    unit.cancel_scope.cancel()
    _flush_queue(unit.input_queue)
    _flush_queue(unit.text_prompt_queue)
    _flush_queue(unit.output_queue, preserve=preserve, on_discard=on_discard)
    _flush_queue(unit.text_output_queue, preserve=preserve, on_discard=on_discard)
    unit.response_playing.clear()
    unit.cancel_scope.reset()
    unit.should_listen.set()


def _to_audio_bytes(chunk: Any) -> bytes:
    chunk = _audio_payload(chunk)
    if isinstance(chunk, PipelineControlMessage):
        raise TypeError(f"unexpected control message on audio output queue: {chunk!r}")
    if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
        return chunk.tobytes()
    if not isinstance(chunk, bytes):
        raise TypeError(f"unexpected audio output type: {type(chunk).__name__}")
    return chunk


def _is_audio_done(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == AUDIO_RESPONSE_DONE


def _is_pipeline_end(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == PIPELINE_END


def _generation_is_discardable(unit: PipelineUnit, generation: int | None) -> bool:
    """Whether output belongs to a cancelled or superseded generation."""

    if generation is not None and unit.cancel_scope.is_stale(generation):
        return True
    if unit.cancel_scope.discarding and generation != unit.cancel_scope.generation:
        return True
    return False


def _should_discard_audio(unit: PipelineUnit, item: Any) -> bool:
    return _generation_is_discardable(unit, _audio_generation(item))


def _safe_unregister(unit: PipelineUnit, session_id: str) -> None:
    try:
        unit.service.unregister(session_id)
    except Exception as exc:
        Logger.error(
            f"Pipeline {unit.index}: unregister failed for session {session_id}: "
            f"{type(exc).__name__}: {exc}"
        )


class SessionRuntime:
    """Own one active client session on an isolated :class:`PipelineUnit`.

    The runtime may be started once.  Input audio and client events can then be
    fed from any transport adapter.  Output is drained by an asyncio task and
    delivered through ``sink``.  ``close`` propagates a tagged ``SESSION_END``
    through every handler before releasing connection state, preventing late
    work from leaking into a later user of the unit.
    """

    def __init__(
        self,
        unit: PipelineUnit,
        sink: SessionSink,
        stop_event: ThreadingEvent,
        *,
        drain_warning_timeout_s: float = SESSION_END_DRAIN_TIMEOUT_S,
        quarantine_timeout_s: float = SESSION_END_QUARANTINE_TIMEOUT_S,
    ) -> None:
        self.unit = unit
        self.sink = sink
        self.stop_event = stop_event
        self.drain_warning_timeout_s = drain_warning_timeout_s
        self.quarantine_timeout_s = quarantine_timeout_s

        self._started = False
        self._closed = False
        self._send_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed_event = asyncio.Event()
        self._delivery_failure: Exception | None = None

    @property
    def session_id(self) -> str | None:
        session = self.unit.session
        if session is None or not session.session_id:
            return None
        return session.session_id

    @property
    def active(self) -> bool:
        session = self.unit.session
        return (
            self._started
            and not self._closed
            and session is not None
            and session.sink is self.sink
            and session.released_at is None
        )

    @property
    def terminal_error(self) -> Exception | None:
        """Return the output-delivery failure that terminated this session."""

        return self._delivery_failure

    @property
    def closed(self) -> bool:
        """Whether cleanup has released all session-owned state."""

        return self._closed

    async def wait_closed(self) -> None:
        """Wait until cleanup has released the pipeline unit and sink."""

        await self._closed_event.wait()

    def _ensure_close_task(self) -> asyncio.Task[None]:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(
                self._close_impl(),
                name=f"s2s-session-close-{self.unit.index}",
            )
            self._close_task = task
        return task

    def fail_delivery(self, exc: Exception) -> None:
        """Make an output failure terminal and start session cleanup once."""

        if self._closed or self._delivery_failure is not None:
            return
        self._delivery_failure = exc
        task = self._ensure_close_task()

        def report_background_close(done: asyncio.Task[None]) -> None:
            if done.cancelled():
                if self._close_task is done:
                    self._close_task = None
                Logger.error(f"Pipeline {self.unit.index}: terminal session cleanup was cancelled")
                return
            close_error = done.exception()
            if close_error is not None:
                if self._close_task is done:
                    self._close_task = None
                Logger.error(
                    f"Pipeline {self.unit.index}: terminal session cleanup failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )

        task.add_done_callback(report_background_close)

    async def start(self) -> str:
        """Register the session, reset pipeline edges, and start output delivery."""

        if self._started:
            raise RuntimeError("SessionRuntime can only be started once")
        if self.unit.session is not None:
            raise RuntimeError(f"Pipeline {self.unit.index} is already in use")

        pipeline_log_ctx.set(self.unit.index)
        session = SessionState(sink=self.sink)
        self.unit.session = session
        session_id = ""
        try:
            session_id = self.unit.service.register()
            session.session_id = session_id
            _clean_unit(self.unit)
            await self.sink.send_events([self.unit.service.build_session_created(session_id)])
            self._send_task = asyncio.create_task(
                self._send_loop(),
                name=f"s2s-session-output-{self.unit.index}",
            )
            self._started = True
            Logger.info(f"Session {session_id} started on pipeline {self.unit.index}")
            return session_id
        except BaseException:
            if self._send_task is not None:
                self._send_task.cancel()
            if session_id:
                _safe_unregister(self.unit, session_id)
            if self.unit.session is session:
                self.unit.session = None
            raise

    def _active_session(self) -> SessionState:
        session = self.unit.session
        if not self.active or session is None:
            raise RuntimeError("SessionRuntime is not active")
        return session

    def feed_pcm(self, pcm: bytes | bytearray | memoryview, sample_rate: int = PIPELINE_SAMPLE_RATE) -> int:
        """Feed raw mono PCM16 into the VAD input and return queued chunk count.

        ``append_pcm`` is a zero-resample path when ``sample_rate`` is the
        pipeline rate.  Any transport-specific conversion should happen before
        this boundary.
        """

        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        session = self._active_session()
        chunks = self.unit.service.append_pcm(session.session_id, bytes(pcm), sample_rate)
        runtime_config = self.unit.service._state(session.session_id).runtime_config
        for chunk in chunks:
            self.unit.input_queue.put((chunk, runtime_config))
        return len(chunks)

    async def handle_event(self, raw: Mapping[str, object]) -> None:
        """Parse and apply one OpenAI-Realtime-shaped control event.

        Audio append events are intentionally rejected: raw audio enters via
        :meth:`feed_pcm`.  Other supported events retain the original realtime
        service's ordering, correlation, and cancellation semantics.
        """

        session = self._active_session()
        service = self.unit.service
        session_id = session.session_id
        client_event_id = raw.get("event_id")

        async def send_correlated(events: list[Any]) -> None:
            if isinstance(client_event_id, str):
                for outgoing in events:
                    if getattr(outgoing, "type", None) == "error":
                        outgoing.error.event_id = client_event_id
            if events:
                await self.sink.send_events(events)

        event = service.parse_client_event(raw)
        if event is None:
            await send_correlated(
                [service.make_error(f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event")]
            )
            return

        if isinstance(event, InputAudioBufferAppendEvent):
            await send_correlated(
                [
                    service.make_error(
                        "Audio is carried by the session PCM stream; input_audio_buffer.append is not supported.",
                        "invalid_event_for_transport",
                    )
                ]
            )
            return

        if isinstance(event, InputAudioBufferCommitEvent):
            error = service.handle_audio_commit(session_id)
            if error:
                await send_correlated([error])
            return

        if isinstance(event, OutputAudioBufferClearEvent):
            _flush_queue(self.unit.output_queue, preserve=_keep_non_audio_output)
            self.sink.discard_pending_audio()
            return

        if isinstance(event, SessionUpdateEvent):
            error = service.handle_session_update(session_id, event)
            if error:
                await send_correlated([error])
            else:
                await send_correlated([service.build_session_updated(session_id)])
            return

        if isinstance(event, ConversationItemCreateEvent):
            events = service.handle_conversation_item_create(session_id, event)
            if events:
                await send_correlated(events)
            return

        if isinstance(event, ConversationItemTruncateEvent):
            # Raw playback is owned by the sink/client. Cancellation has already
            # invalidated provisional server generation, so no chat mutation is
            # required here.
            Logger.debug(f"Accepted conversation.item.truncate for {event.item_id}")
            return

        if isinstance(event, ResponseCreateEvent):
            result = service.handle_response_create(session_id, event)
            if result:
                response_key = None
                if result.type != "error":
                    self.unit.cancel_scope.new_response()
                    response_key = service._state(session_id).current_response_key
                await send_correlated([result])
                if result.type == "response.created":
                    service.response.mark_response_created_sent(session_id, response_key)
            return

        if isinstance(event, ResponseCancelEvent):
            state = service._state(session_id)
            had_response = state.in_response or state.response_pending
            if had_response:
                self.unit.cancel_scope.cancel()
                _flush_queue(self.unit.text_prompt_queue, preserve=_keep_pipeline_control)
            _flush_queue(self.unit.output_queue, preserve=_keep_cancel_bookkeeping)
            _flush_queue(self.unit.text_output_queue, preserve=_keep_user_text_event)
            self.sink.discard_pending_audio()
            events = service.handle_response_cancel(session_id)
            if events:
                await send_correlated(events)
            self.unit.response_playing.clear()

    async def close(self) -> None:
        """Drain all handlers, unregister service state, and close the sink.

        The cleanup runs in a shielded task.  If the caller is cancelled during
        shutdown, the task retains a strong reference and continues protecting
        the pipeline unit from premature reuse.
        """

        if self._closed:
            return
        close_task = self._ensure_close_task()
        try:
            await asyncio.shield(close_task)
        except BaseException:
            # A failed completed task must not poison every later close call.
            # A caller cancellation leaves the shielded cleanup running.
            if close_task.done() and self._close_task is close_task:
                self._close_task = None
            raise

    async def _close_impl(self) -> None:
        if self._closed:
            self._closed_event.set()
            return
        session = self.unit.session
        owns_session = self._started and session is not None and session.sink is self.sink
        session_id = session.session_id if owns_session and session is not None else ""
        failure: BaseException | None = None

        def remember_failure(stage: str, exc: BaseException) -> None:
            nonlocal failure
            if failure is None:
                failure = exc
            Logger.error(
                f"Pipeline {self.unit.index}: session cleanup {stage} failed: "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            if owns_session and session is not None:
                session.released_at = time.monotonic()

                try:
                    self.unit.service.close_pending_responses(session_id)
                except KeyError:
                    pass
                except BaseException as exc:
                    remember_failure("close_pending_responses", exc)

                def account_usage(item: Any) -> None:
                    if not isinstance(item, TokenUsageEvent):
                        return
                    try:
                        self.unit.service.dispatch_pipeline_event(session_id, item)
                    except KeyError:
                        Logger.debug(f"Skipped late usage for unregistered session {session_id}")
                    except Exception as exc:
                        remember_failure("usage accounting", exc)

                try:
                    if session.pending_output_item is not None:
                        account_usage(session.pending_output_item)
                    for item in session.pending_text_output_items:
                        account_usage(item)
                except BaseException as exc:
                    remember_failure("pending output accounting", exc)
                finally:
                    session.pending_output_item = None
                    session.pending_text_output_items.clear()

                try:
                    _clean_unit(self.unit, on_discard=account_usage)
                except BaseException as exc:
                    remember_failure("pipeline reset", exc)

                drain_requested = False
                try:
                    self.unit.input_queue.put(PipelineControlMessage(SESSION_END.kind, session_id=session_id))
                    drain_requested = True
                except BaseException as exc:
                    remember_failure("SESSION_END enqueue", exc)

                warned = False
                started_waiting = time.monotonic()
                if drain_requested:
                    while not session.drained.is_set() and not self.stop_event.is_set():
                        try:
                            await asyncio.wait_for(session.drained.wait(), timeout=0.05)
                        except asyncio.TimeoutError:
                            pass
                        elapsed = time.monotonic() - started_waiting
                        if not warned and elapsed >= self.drain_warning_timeout_s:
                            Logger.warning(
                                f"Pipeline {self.unit.index}: SESSION_END not drained after "
                                f"{elapsed:.1f}s; unit remains unavailable"
                            )
                            warned = True
                        if session.quarantined_at is None and elapsed >= self.quarantine_timeout_s:
                            session.quarantined_at = time.monotonic()
                            _safe_unregister(self.unit, session_id)
                            Logger.error(
                                f"Pipeline {self.unit.index}: SESSION_END still not drained after "
                                f"{elapsed:.0f}s; quarantining until drain"
                            )

                    if self.stop_event.is_set() and not session.drained.is_set():
                        Logger.info(
                            f"Pipeline {self.unit.index} stopped before SESSION_END drained "
                            f"for session {session_id}"
                        )
        except BaseException as exc:
            remember_failure("drain", exc)
        finally:
            if session_id:
                _safe_unregister(self.unit, session_id)
            if owns_session and self.unit.session is session:
                self.unit.session = None

            send_task = self._send_task
            current_task = asyncio.current_task()
            if send_task is not None and send_task is not current_task and not send_task.done():
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    remember_failure("output task", exc)
            try:
                await self.sink.close()
            except BaseException as exc:
                remember_failure("sink close", exc)
            finally:
                self._closed = True
                self._closed_event.set()

            if owns_session and session is not None:
                recovered = " after quarantine" if session.quarantined_at is not None else ""
                Logger.info(f"Pipeline {self.unit.index} released{recovered} (session {session_id} ended)")

        if failure is not None:
            raise failure

    async def _drain_after_delivery_failure(self, session: SessionState) -> None:
        """Discard output until SESSION_END without touching the failed sink."""

        unit = self.unit
        while unit.session is session and not self.stop_event.is_set():
            _flush_queue(unit.text_output_queue)
            try:
                item = unit.output_queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.01)
                continue

            if is_control_message(item, SESSION_END.kind):
                item_session_id = getattr(item, "session_id", None)
                if item_session_id in (None, session.session_id):
                    session.drained.set()
                    Logger.debug(f"Pipeline {unit.index}: failed session SESSION_END drained")
                    return
            if _is_pipeline_end(item):
                session.drained.set()
                return

    async def _send_loop(self) -> None:
        """Drain ordered pipeline output and deliver it through the sink."""

        unit = self.unit
        pipeline_log_ctx.set(unit.index)
        while not self.stop_event.is_set():
            try:
                session = unit.session
                if self._delivery_failure is not None:
                    if session is not None:
                        await self._drain_after_delivery_failure(session)
                    break
                sink = session.sink if session is not None else None
                session_id = session.session_id if session is not None else None

                # Side-channel text/events are checked first because speech
                # start must cancel active output before another audio batch is
                # exposed.
                try:
                    text_message = None
                    if session is not None and session_id is not None:
                        for index, pending in enumerate(session.pending_text_output_items):
                            if not _response_key_output_is_blocked(
                                unit,
                                session_id,
                                _output_response_key(pending),
                            ):
                                text_message = session.pending_text_output_items.pop(index)
                                break
                    if text_message is None:
                        text_message = unit.text_output_queue.get_nowait()

                    if (
                        session is not None
                        and session_id is not None
                        and _response_key_output_is_blocked(
                            unit,
                            session_id,
                            _output_response_key(text_message),
                        )
                    ):
                        session.pending_text_output_items.append(text_message)
                        text_message = None
                    if text_message is None:
                        raise Empty

                    if isinstance(text_message, AssistantToolCallReadyEvent):
                        generation = text_message.cancel_generation
                        response_key = text_message.response_key
                        if _generation_is_discardable(unit, generation):
                            continue
                        if session_id is not None and _response_key_is_obsolete(unit, session_id, response_key):
                            _discard_obsolete_response_key(unit, session_id, response_key)
                            continue

                    is_speech_start = isinstance(text_message, SpeechStartedEvent)
                    was_in_response = False
                    was_response_pending = False
                    if is_speech_start and session_id:
                        state = unit.service._state(session_id)
                        was_in_response = state.in_response
                        was_response_pending = state.response_pending

                    if sink is not None and isinstance(text_message, PipelineEvent) and session_id:
                        events = unit.service.dispatch_pipeline_event(session_id, text_message)
                        if events:
                            await sink.send_events(events)

                    if isinstance(text_message, SpeechStartedEvent) and session_id:
                        active_config = unit.service._state(session_id).runtime_config
                        interrupt_enabled = text_message.interrupt_response and (
                            active_config is None or active_config.interrupt_response_enabled
                        )
                        if interrupt_enabled and sink is not None:
                            sink.discard_pending_audio()
                        if was_in_response or was_response_pending:
                            if interrupt_enabled:
                                unit.cancel_scope.cancel()
                                unit.service.close_pending_responses(session_id)
                                _flush_queue(unit.text_prompt_queue, preserve=_keep_pipeline_control)
                                _flush_queue(unit.output_queue, preserve=_keep_cancel_bookkeeping)
                                _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                                if unit.response_playing.is_set():
                                    unit.response_playing.clear()
                                response_state = "response" if was_in_response else "pending response"
                                Logger.info(
                                    f"Pipeline {unit.index}: speech during {response_state}: "
                                    "cancelled, queue flushed"
                                )
                            else:
                                Logger.info(
                                    f"Pipeline {unit.index}: speech during response: "
                                    "interrupt_response disabled, ignoring"
                                )
                except Empty:
                    pass

                try:
                    if session is not None and session.pending_output_item is not None:
                        audio_chunk = session.pending_output_item
                        session.pending_output_item = None
                    else:
                        audio_chunk = unit.output_queue.get_nowait()

                    if (
                        session is not None
                        and session_id is not None
                        and _response_key_output_is_blocked(
                            unit,
                            session_id,
                            _output_response_key(audio_chunk),
                        )
                    ):
                        session.pending_output_item = audio_chunk
                        await asyncio.sleep(0.01)
                        continue

                    if isinstance(audio_chunk, TokenUsageEvent):
                        if sink is not None and session_id is not None:
                            events = unit.service.dispatch_pipeline_event(session_id, audio_chunk)
                            if events:
                                await sink.send_events(events)
                        continue

                    if isinstance(audio_chunk, _RESPONSE_PIPELINE_EVENTS):
                        generation = getattr(audio_chunk, "cancel_generation", None)
                        response_key = _response_event_key(audio_chunk)
                        if _generation_is_discardable(unit, generation):
                            continue
                        if session_id is not None and _response_key_is_obsolete(unit, session_id, response_key):
                            _discard_obsolete_response_key(unit, session_id, response_key)
                            continue
                        if sink is not None and session_id is not None:
                            events = unit.service.dispatch_pipeline_event(session_id, audio_chunk)
                            if events:
                                await sink.send_events(events)
                        continue

                    if _is_pipeline_end(audio_chunk):
                        if sink is not None and session_id:
                            events = unit.service.finish_response(session_id)
                            if events:
                                await sink.send_events(events)
                        break

                    if _is_audio_done(audio_chunk):
                        audio_generation = _audio_generation(audio_chunk)
                        response_key = _audio_response_key(audio_chunk)
                        if _audio_cleanup_only(audio_chunk):
                            if response_key is None:
                                Logger.warning("Ignoring unkeyed stale response cleanup terminal")
                                continue
                            cleaned_active_response = False
                            if session_id:
                                state = unit.service._state(session_id)
                                if state.in_response and state.current_response_key in (None, response_key):
                                    cleaned_active_response = True
                                    events = unit.service.finish_response(
                                        session_id,
                                        status="cancelled",
                                        response_key=response_key,
                                    )
                                    if sink is not None and events:
                                        await sink.send_events(events)
                                else:
                                    unit.service.close_response_key(session_id, response_key)
                                if cleaned_active_response:
                                    unit.response_playing.clear()
                                if not (state.in_response or state.response_pending):
                                    unit.should_listen.set()
                            unit.cancel_scope.response_done(audio_generation)
                            Logger.info(f"Pipeline {unit.index}: stale response lifecycle cleaned up")
                            continue

                        if audio_generation is not None and unit.cancel_scope.is_stale(audio_generation):
                            if session_id:
                                unit.service.close_response_key(session_id, response_key)
                            unit.cancel_scope.response_done(audio_generation)
                            unit.should_listen.set()
                            Logger.info(f"Pipeline {unit.index}: stale response complete, listening re-enabled")
                            continue

                        if session_id is not None and _response_key_is_obsolete(unit, session_id, response_key):
                            _discard_obsolete_response_key(unit, session_id, response_key)
                            continue

                        if sink is not None and session_id:
                            events = unit.service.finish_response(session_id, response_key=response_key)
                            if events:
                                await sink.send_events(events)
                        if session_id:
                            unit.service._state(session_id).clear_pending_response(response_key)
                        unit.response_playing.clear()
                        unit.cancel_scope.response_done(audio_generation)
                        unit.should_listen.set()
                        Logger.info(f"Pipeline {unit.index}: response complete, listening re-enabled")
                        continue

                    if is_control_message(audio_chunk, SESSION_END.kind):
                        chunk_session_id = getattr(audio_chunk, "session_id", None)
                        if session is not None and chunk_session_id in (None, session.session_id):
                            session.drained.set()
                            Logger.debug(f"Pipeline {unit.index}: SESSION_END drained")
                        continue

                    if is_control_message(audio_chunk):
                        continue

                    if _should_discard_audio(unit, audio_chunk):
                        continue

                    response_key = _audio_response_key(audio_chunk)
                    if session_id is not None and _response_key_is_obsolete(unit, session_id, response_key):
                        _discard_obsolete_response_key(unit, session_id, response_key)
                        continue

                    audio_batch = bytearray(_to_audio_bytes(audio_chunk))
                    while len(audio_batch) < MAX_AUDIO_BATCH_BYTES:
                        try:
                            next_chunk = unit.output_queue.get_nowait()
                        except Empty:
                            break

                        if (
                            _is_pipeline_end(next_chunk)
                            or _is_audio_done(next_chunk)
                            or isinstance(next_chunk, PipelineEvent)
                            or is_control_message(next_chunk, SESSION_END.kind)
                        ):
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break

                        if _should_discard_audio(unit, next_chunk):
                            continue
                        if _audio_response_key(next_chunk) != response_key:
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break

                        next_audio = _to_audio_bytes(next_chunk)
                        if len(audio_batch) + len(next_audio) > MAX_AUDIO_BATCH_BYTES:
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break
                        audio_batch.extend(next_audio)

                    if not unit.response_playing.is_set():
                        unit.response_playing.set()
                        unit.should_listen.set()

                    if sink is not None and session_id:
                        # Response/item bookkeeping belongs to the runtime, not
                        # to a media transport. This also guarantees that an
                        # implicit response.created is delivered before PCM.
                        _response_id, _item_id, _output_index, events = unit.service.begin_audio_output(
                            session_id,
                            response_key,
                        )
                        if events:
                            await sink.send_events(events)
                        await sink.send_audio(bytes(audio_batch), response_key)
                except Empty:
                    pass

                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                Logger.error(
                    f"Pipeline {unit.index} send loop terminal error: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.fail_delivery(exc)
                # Let the close task reset the queues and enqueue SESSION_END;
                # the next iteration switches to drain-only mode.
                await asyncio.sleep(0)


__all__ = [
    "MAX_AUDIO_BATCH_BYTES",
    "SESSION_END_DRAIN_TIMEOUT_S",
    "SESSION_END_QUARANTINE_TIMEOUT_S",
    "SessionRuntime",
]
