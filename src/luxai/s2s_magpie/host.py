"""MAGPIE host for reusable transport-neutral S2S sessions."""

from __future__ import annotations

import asyncio
from threading import Event
from typing import Any

from luxai.magpie.frames import AudioFrameRaw, DictFrame, Frame
from luxai.magpie.nodes import ServerNode
from luxai.magpie.transport import ZMQRpcResponder
from luxai.magpie.utils import Logger
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.runtime.session import SessionRuntime

from .protocol import (
    AUDIO_INPUT_PORT_OFFSET,
    AUDIO_INPUT_TOPIC,
    AUDIO_OUTPUT_PORT_OFFSET,
    EVENT_INPUT_PORT_OFFSET,
    EVENT_INPUT_TOPIC,
    EVENT_OUTPUT_PORT_OFFSET,
    FrameId,
    RPC_PORT_OFFSET,
    STATUS_RPC,
    bind_endpoint,
    build_system_descriptor,
)
from .transport import MagpieSessionSink, MagpieSinkError, StrictZmqStreamReader


def _decode_frame(raw: object) -> Frame:
    if isinstance(raw, Frame):
        return raw
    if isinstance(raw, dict):
        return Frame.from_dict(raw)
    raise TypeError(f"Expected a MAGPIE frame, got {type(raw).__name__}")


class MagpieSessionHost:
    """Bridge native MAGPIE streams to one reusable S2S pipeline unit.

    MAGPIE streams do not have the connection lifecycle that a WebSocket has.
    The first client event therefore opens a fresh S2S session implicitly, and
    ``magpie.session.close`` closes it explicitly. A later client event can
    open another clean session without reloading the model handlers.
    """

    def __init__(
        self,
        unit: PipelineUnit,
        stop_event: Event,
        params: Any,
    ) -> None:
        self._unit = unit
        self._stop_event = stop_event
        self._base_port = int(params.zmq.port)
        self._node_id = str(params.zmq.node_id)
        self._audio_input_queue_size = int(params.zmq.audio_input_queue_size)
        self._audio_output_queue_size = int(params.zmq.audio_output_queue_size)
        self._event_input_queue_size = int(params.zmq.event_input_queue_size)
        self._event_output_queue_size = int(params.zmq.event_output_queue_size)
        self._descriptor = build_system_descriptor(
            self._node_id,
            self._base_port,
            audio_input_queue_size=self._audio_input_queue_size,
            audio_output_queue_size=self._audio_output_queue_size,
            event_input_queue_size=self._event_input_queue_size,
            event_output_queue_size=self._event_output_queue_size,
        )
        self._runtime: SessionRuntime | None = None
        self._runtime_watch_task: asyncio.Task[None] | None = None
        self._client_gid: FrameId | None = None
        self._closing_client_gid: FrameId | None = None
        self._session_close_task: asyncio.Task[None] | None = None
        self._pending_session_update: tuple[FrameId, dict[str, Any]] | None = None
        self._retired_client_gids: dict[FrameId, None] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._stopped = asyncio.Event()
        self._fatal_error: Exception | None = None
        self._fatal_event = asyncio.Event()
        self._stop_task: asyncio.Task[None] | None = None
        self._stop_complete = False
        self._audio_reader_closed = False
        self._event_reader_closed = False
        self._sink_shutdown = False
        self._rpc_terminated = False

        # Follow the standard LuxAI driver layout: the RPC responder occupies
        # the configured base port and every stream is a fixed offset from it.
        # ServerNode starts its dedicated RPC thread during construction.
        self._rpc_server = ServerNode(
            ZMQRpcResponder(bind_endpoint(self._base_port, RPC_PORT_OFFSET)),
            self._on_rpc,
            name="s2s-magpie-rpc",
        )
        self._sink = MagpieSessionSink(
            audio_endpoint=bind_endpoint(self._base_port, AUDIO_OUTPUT_PORT_OFFSET),
            event_endpoint=bind_endpoint(self._base_port, EVENT_OUTPUT_PORT_OFFSET),
            audio_queue_size=self._audio_output_queue_size,
            event_queue_size=self._event_output_queue_size,
        )
        self._audio_reader = StrictZmqStreamReader(
            bind_endpoint(self._base_port, AUDIO_INPUT_PORT_OFFSET),
            topic=AUDIO_INPUT_TOPIC,
            queue_size=self._audio_input_queue_size,
            bind=True,
            delivery="reliable",
        )
        self._event_reader = StrictZmqStreamReader(
            bind_endpoint(self._base_port, EVENT_INPUT_PORT_OFFSET),
            topic=EVENT_INPUT_TOPIC,
            queue_size=self._event_input_queue_size,
            bind=True,
            delivery="reliable",
        )

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._audio_input_loop(), name="magpie-s2s-audio-input"),
            asyncio.create_task(self._event_input_loop(), name="magpie-s2s-event-input"),
        ]
        Logger.info(
            "MAGPIE S2S host ready; "
            f"RPC={self._base_port}, audio IN={self._base_port + AUDIO_INPUT_PORT_OFFSET}, "
            f"audio OUT={self._base_port + AUDIO_OUTPUT_PORT_OFFSET}, "
            f"events IN={self._base_port + EVENT_INPUT_PORT_OFFSET}, "
            f"events OUT={self._base_port + EVENT_OUTPUT_PORT_OFFSET}"
        )

    async def wait(self) -> None:
        if not self._tasks:
            raise RuntimeError("MAGPIE S2S host has not been started")
        fatal_waiter = asyncio.create_task(
            self._fatal_event.wait(),
            name="magpie-s2s-fatal-output-wait",
        )
        try:
            done, _ = await asyncio.wait(
                [*self._tasks, fatal_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if fatal_waiter in done:
                error = self._fatal_error
                if error is None:
                    raise RuntimeError("MAGPIE S2S host entered a fatal state")
                raise error
            for task in done:
                task.result()
        finally:
            fatal_waiter.cancel()
            await asyncio.gather(fatal_waiter, return_exceptions=True)

    def _fail_output_transport(self, exc: Exception) -> None:
        if self._fatal_error is None:
            self._fatal_error = exc
            self._fatal_event.set()

    async def stop(self) -> None:
        if self._stop_complete:
            return
        stop_task = self._stop_task
        if stop_task is None:
            stop_task = asyncio.create_task(self._stop_impl(), name="magpie-s2s-host-cleanup")
            self._stop_task = stop_task
        try:
            await asyncio.shield(stop_task)
        except BaseException:
            if stop_task.done() and self._stop_task is stop_task:
                self._stop_task = None
            raise

    async def _stop_impl(self) -> None:
        failures: list[BaseException] = []

        def remember_failure(resource: str, exc: BaseException) -> None:
            failures.append(exc)
            Logger.error(
                f"Failed to stop MAGPIE S2S {resource}: "
                f"{type(exc).__name__}: {exc}"
            )

        self._stopped.set()
        for task in self._tasks:
            task.cancel()

        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except BaseException as exc:
            remember_failure("input tasks", exc)
        finally:
            self._tasks.clear()

        close_task = self._session_close_task
        if close_task is not None:
            try:
                await asyncio.shield(close_task)
            except BaseException as exc:
                remember_failure("session drain task", exc)

        try:
            await self._close_session()
        except BaseException as exc:
            remember_failure("active session", exc)

        runtime_watch = self._runtime_watch_task
        if runtime_watch is not None and not runtime_watch.done():
            runtime_watch.cancel()
        if runtime_watch is not None:
            try:
                await runtime_watch
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                remember_failure("runtime watcher", exc)
            finally:
                if self._runtime_watch_task is runtime_watch:
                    self._runtime_watch_task = None

        if not self._audio_reader_closed:
            try:
                await asyncio.to_thread(self._audio_reader.close)
            except BaseException as exc:
                remember_failure("audio reader", exc)
            else:
                self._audio_reader_closed = True

        if not self._event_reader_closed:
            try:
                await asyncio.to_thread(self._event_reader.close)
            except BaseException as exc:
                remember_failure("event reader", exc)
            else:
                self._event_reader_closed = True

        if not self._sink_shutdown:
            try:
                await self._sink.shutdown()
            except BaseException as exc:
                remember_failure("output sink", exc)
            else:
                self._sink_shutdown = True

        if not self._rpc_terminated:
            try:
                await asyncio.to_thread(self._rpc_server.terminate, 1.0)
            except BaseException as exc:
                remember_failure("RPC server", exc)
            else:
                self._rpc_terminated = True

        self._stop_complete = (
            self._audio_reader_closed
            and self._event_reader_closed
            and self._sink_shutdown
            and self._rpc_terminated
            and self._runtime is None
        )
        if failures:
            raise failures[0]

    def _on_rpc(self, request: dict) -> dict:
        """Serve discovery and lightweight process/session status."""

        if not isinstance(request, dict):
            return {"status": False, "response": "invalid request"}

        name = request.get("name", "")
        if not name:
            return {"status": True, "response": self._descriptor}

        if name == STATUS_RPC:
            runtime = self._runtime
            return {
                "status": True,
                "response": {
                    "running": any(not task.done() for task in self._tasks)
                    and not self._stopped.is_set(),
                    "session_active": bool(runtime is not None and runtime.active),
                    "client_gid": self._client_gid,
                },
            }

        Logger.warning(f"Unknown S2S RPC service: {name}")
        return {"status": False, "response": f"unknown service: {name}"}

    async def _audio_input_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                result = await asyncio.to_thread(self._audio_reader.read, 0.25)
            except TimeoutError:
                continue
            if result is None:
                continue
            raw, _topic = result
            try:
                frame = _decode_frame(raw)
                if not isinstance(frame, AudioFrameRaw):
                    raise TypeError(f"Expected AudioFrameRaw, got {type(frame).__name__}")
                if not frame.data:
                    continue
                if frame.format.upper() != "PCM" or frame.bit_depth != 16 or frame.channels != 1:
                    raise ValueError(
                        "S2S input requires mono PCM16; "
                        f"received {frame.format}, {frame.channels} channel(s), {frame.bit_depth}-bit"
                    )
                runtime = self._runtime
                if (
                    runtime is None
                    or not runtime.active
                    or frame.gid != self._client_gid
                ):
                    continue
                runtime.feed_pcm(bytes(frame.data), sample_rate=frame.sample_rate)
            except Exception as exc:
                Logger.warning(f"Discarding invalid MAGPIE audio frame: {exc}")

    async def _event_input_loop(self) -> None:
        while not self._stopped.is_set():
            event_gid: FrameId | None = None
            try:
                result = await asyncio.to_thread(self._event_reader.read, 0.25)
            except TimeoutError:
                continue
            if result is None:
                continue
            raw, _topic = result
            try:
                frame = _decode_frame(raw)
                if not isinstance(frame, DictFrame):
                    raise TypeError(f"Expected DictFrame, got {type(frame).__name__}")
                event_gid = frame.gid
                if not isinstance(frame.value, dict):
                    raise TypeError("MAGPIE event value must be a dictionary")
                event_type = frame.value.get("type")
                if event_type == "magpie.session.close":
                    if frame.gid in {self._client_gid, self._closing_client_gid}:
                        await self._sink.send_session_closing(frame.gid)
                        if self._session_close_task is None:
                            self._closing_client_gid = frame.gid
                            self._session_close_task = asyncio.create_task(
                                self._drain_session(frame.gid),
                                name=f"magpie-s2s-session-close-{frame.gid}",
                            )
                    else:
                        # Closing an already retired or unknown session is
                        # idempotent and must not make a client wait for a
                        # timeout after reconnect/preemption.
                        pending = self._pending_session_update
                        if pending is not None and pending[0] == frame.gid:
                            self._pending_session_update = None
                        await self._sink.send_session_closed(frame.gid)
                    continue
                if self._session_close_task is not None:
                    if (
                        event_type == "session.update"
                        and frame.gid not in self._retired_client_gids
                        and frame.gid
                        not in {self._client_gid, self._closing_client_gid}
                    ):
                        # Keep only the latest configuration for the next
                        # client. Audio remains gated on session.updated, so no
                        # media can cross the session boundary while we drain.
                        self._pending_session_update = (
                            frame.gid,
                            dict(frame.value),
                        )
                        Logger.debug(
                            "Deferred S2S session.update until prior drain: "
                            f"gid={frame.gid}"
                        )
                    else:
                        Logger.debug(
                            "Ignoring S2S event while the prior session drains: "
                            f"type={event_type}, gid={frame.gid}"
                        )
                    continue
                if frame.gid in self._retired_client_gids:
                    continue
                runtime = self._runtime
                if (
                    (runtime is None or frame.gid != self._client_gid)
                    and event_type != "session.update"
                ):
                    Logger.debug(
                        "Ignoring S2S event without active session ownership: "
                        f"type={event_type}, gid={frame.gid}"
                    )
                    continue
                runtime = await self._ensure_session(frame.gid)
                await runtime.handle_event(frame.value)
            except MagpieSinkError as exc:
                Logger.error(f"MAGPIE output failed; terminating client session: {exc}")
                self._fail_output_transport(exc)
                runtime = self._runtime
                if event_gid == self._client_gid and runtime is not None:
                    runtime.fail_delivery(exc)
            except Exception as exc:
                Logger.error(f"MAGPIE client event failed: {exc}")
                if event_gid == self._client_gid:
                    try:
                        await self._sink.send_events(
                            [
                                {
                                    "type": "error",
                                    "error": {
                                        "type": "magpie_event_error",
                                        "message": str(exc),
                                    },
                                }
                            ]
                        )
                    except MagpieSinkError as sink_exc:
                        Logger.error(
                            "MAGPIE error-event delivery failed; terminating "
                            f"client session: {sink_exc}"
                        )
                        self._fail_output_transport(sink_exc)
                        runtime = self._runtime
                        if runtime is not None:
                            runtime.fail_delivery(sink_exc)
                    except Exception as sink_exc:
                        Logger.error(f"MAGPIE error-event delivery failed: {sink_exc}")

    async def _ensure_session(self, client_gid: FrameId) -> SessionRuntime:
        runtime = self._runtime
        if runtime is not None and runtime.active and self._client_gid == client_gid:
            return runtime
        if runtime is not None:
            await self._close_session()
        self._sink.open_session(client_gid)
        runtime = SessionRuntime(self._unit, self._sink, self._stop_event)
        try:
            await runtime.start()
        except BaseException:
            try:
                await self._sink.close()
            except Exception as close_exc:
                Logger.error(f"Failed to reset MAGPIE sink after session start failure: {close_exc}")
                if isinstance(close_exc, MagpieSinkError):
                    self._fail_output_transport(close_exc)
            raise
        self._runtime = runtime
        self._client_gid = client_gid
        self._runtime_watch_task = asyncio.create_task(
            self._watch_runtime(runtime, client_gid),
            name=f"magpie-s2s-runtime-watch-{client_gid}",
        )
        Logger.info(f"MAGPIE client session opened: gid={client_gid}")
        return runtime

    async def _watch_runtime(self, runtime: SessionRuntime, client_gid: FrameId) -> None:
        """Detach a runtime that closed itself after terminal delivery failure."""

        this_task = asyncio.current_task()
        try:
            await runtime.wait_closed()
            self._detach_runtime(runtime, client_gid)
            if runtime.terminal_error is not None:
                Logger.error(
                    "MAGPIE client session released after terminal output failure: "
                    f"gid={client_gid}, error={runtime.terminal_error}"
                )
                self._fail_output_transport(runtime.terminal_error)
        finally:
            if self._runtime_watch_task is this_task:
                self._runtime_watch_task = None

    def _detach_runtime(self, runtime: SessionRuntime, client_gid: FrameId | None) -> bool:
        if self._runtime is not runtime:
            return False
        self._runtime = None
        self._client_gid = None
        if client_gid is not None:
            self._retired_client_gids[client_gid] = None
            while len(self._retired_client_gids) > 128:
                self._retired_client_gids.pop(next(iter(self._retired_client_gids)))
        Logger.info(f"MAGPIE client session closed: gid={client_gid}")
        return True

    async def _drain_session(self, client_gid: FrameId) -> None:
        """Drain one accepted close without blocking duplicate acknowledgements."""

        this_task = asyncio.current_task()
        released = False
        try:
            try:
                await self._close_session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                Logger.error(
                    f"MAGPIE client session drain failed: gid={client_gid}, error={exc}"
                )
                if isinstance(exc, MagpieSinkError):
                    self._fail_output_transport(exc)
            finally:
                released = self._runtime is None

            if released:
                try:
                    await self._sink.send_session_closed(client_gid)
                except Exception as exc:
                    Logger.error(
                        "MAGPIE session closed acknowledgement failed: "
                        f"gid={client_gid}, error={exc}"
                    )
                    if isinstance(exc, MagpieSinkError):
                        self._fail_output_transport(exc)
        finally:
            if self._session_close_task is this_task:
                self._session_close_task = None
                self._closing_client_gid = None

        if released and not self._stopped.is_set() and not self._stop_event.is_set():
            pending = self._pending_session_update
            self._pending_session_update = None
            if pending is not None:
                pending_gid, pending_event = pending
                try:
                    runtime = await self._ensure_session(pending_gid)
                    await runtime.handle_event(pending_event)
                except Exception as exc:
                    Logger.error(
                        "Deferred MAGPIE session.update failed: "
                        f"gid={pending_gid}, error={exc}"
                    )
                    if isinstance(exc, MagpieSinkError):
                        self._fail_output_transport(exc)
                        runtime = self._runtime
                        if runtime is not None:
                            runtime.fail_delivery(exc)

    async def _close_session(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        client_gid = self._client_gid
        try:
            await runtime.close()
        finally:
            # Runtime cleanup may deliberately re-raise a sink failure after it
            # has released the pipeline. Do not retain that completed runtime.
            # If caller cancellation interrupted the shielded wait, its watcher
            # will detach it once background cleanup actually completes.
            if runtime.closed:
                self._detach_runtime(runtime, client_gid)
