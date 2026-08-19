"""Native MAGPIE output sink for the S2S session runtime."""

from __future__ import annotations

from collections.abc import Mapping
from queue import Empty
from typing import Any

from luxai.magpie.frames import DictFrame
from luxai.magpie.transport import ZmqStreamReader, ZmqStreamWriter
from luxai.magpie.utils import Logger
from luxai.magpie.utils.common import get_uinque_id
from pydantic import BaseModel

from .protocol import (
    AUDIO_OUTPUT_TOPIC,
    EVENT_OUTPUT_TOPIC,
    PIPELINE_BIT_DEPTH,
    PIPELINE_CHANNELS,
    PIPELINE_SAMPLE_RATE,
    FrameId,
    S2SAudioFrame,
)


class MagpieSinkError(RuntimeError):
    """A terminal failure while publishing or closing MAGPIE output."""


class StrictZmqStreamWriter(ZmqStreamWriter):
    """Direct MAGPIE ZMQ writer whose transport errors are not swallowed."""

    def __init__(self, *args: Any, queue_size: int = 0, **kwargs: Any) -> None:
        if queue_size != 0:
            raise ValueError("StrictZmqStreamWriter requires queue_size=0")
        super().__init__(*args, queue_size=queue_size, **kwargs)

    def _transport_write(self, data: object, topic: str | None) -> None:
        topic_bytes = (topic or "").encode()
        payload = self.serializer.serialize(data)
        self.socket.send_multipart([topic_bytes, memoryview(payload)], copy=False)

    def write(self, data: object, topic: str | None = None) -> None:
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")
        self._transport_write(data, topic)

    def close(self) -> None:
        if self._closed:
            return
        failures: list[Exception] = []
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            try:
                self.socket.disable_monitor()
                monitor.close(linger=0)
            except Exception as exc:
                failures.append(exc)
            finally:
                self._monitor = None
        try:
            self.socket.close(linger=0)
        except Exception as exc:
            failures.append(exc)
        if not self.endpoint.startswith("inproc:"):
            try:
                self.context.term()
            except Exception as exc:
                failures.append(exc)
        self._closed = not failures
        if failures:
            raise RuntimeError(f"{self.name} close failed: {failures[0]}") from failures[0]


class StrictZmqStreamReader(ZmqStreamReader):
    """Queued MAGPIE reader that exposes a terminal background transport error."""

    def __init__(self, *args: Any, queue_size: int, **kwargs: Any) -> None:
        if queue_size <= 0:
            raise ValueError("StrictZmqStreamReader requires a positive queue_size")
        self._terminal_error: Exception | None = None
        super().__init__(*args, queue_size=queue_size, **kwargs)

    def _read_thread(self) -> None:
        while not self.reader_close_event.is_set():
            try:
                raw_data = self._transport_read_blocking(timeout=1.0)
                if raw_data is None:
                    continue
                if self.reader_queue.full():
                    try:
                        self.reader_queue.get_nowait()
                    except Empty:
                        pass
                self.reader_queue.put(raw_data)
            except TimeoutError:
                continue
            except Exception as exc:
                self._terminal_error = exc
                self.reader_close_event.set()
                return

    def read(self, timeout: float | None = None):
        if self._terminal_error is not None:
            raise RuntimeError(f"{self.name} transport failed: {self._terminal_error}") from self._terminal_error
        result = super().read(timeout=timeout)
        if self._terminal_error is not None:
            raise RuntimeError(f"{self.name} transport failed: {self._terminal_error}") from self._terminal_error
        return result

    def close(self) -> None:
        if self._closed:
            return
        self.reader_close_event.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError(f"{self.name} reader thread did not stop")

        failures: list[Exception] = []
        try:
            self.socket.close(linger=0)
        except Exception as exc:
            failures.append(exc)
        if not self.endpoint.startswith("inproc:"):
            try:
                self.context.term()
            except Exception as exc:
                failures.append(exc)
        self._closed = not failures
        if failures:
            raise RuntimeError(f"{self.name} close failed: {failures[0]}") from failures[0]


def _event_payload(event: Any) -> dict[str, Any]:
    if isinstance(event, BaseModel):
        return event.model_dump(mode="json", exclude_none=True)
    if isinstance(event, Mapping):
        return dict(event)
    raise TypeError(f"Unsupported S2S event type: {type(event).__name__}")


class MagpieSessionSink:
    """Publish ordered S2S events and raw PCM through MAGPIE.

    One MAGPIE audio GID is used per assistant response. An empty frame with
    that same GID terminates the response, matching QTrobot's streamed-audio
    convention without base64 or JSON audio deltas.
    """

    kind = "magpie"

    def __init__(
        self,
        audio_endpoint: str,
        event_endpoint: str,
        *,
        audio_topic: str = AUDIO_OUTPUT_TOPIC,
        event_topic: str = EVENT_OUTPUT_TOPIC,
        audio_queue_size: int,
        event_queue_size: int,
    ) -> None:
        self._audio_topic = audio_topic
        self._event_topic = event_topic
        # Direct writes preserve ordering and avoid StreamWriter's drop-oldest
        # queue policy, which is unsuitable for PCM and lifecycle events.
        audio_writer = StrictZmqStreamWriter(
            audio_endpoint,
            queue_size=audio_queue_size,
            bind=True,
            delivery="reliable",
        )
        try:
            event_writer = StrictZmqStreamWriter(
                event_endpoint,
                queue_size=event_queue_size,
                bind=True,
                delivery="reliable",
            )
        except BaseException:
            try:
                audio_writer.close()
            except Exception as exc:
                Logger.error(f"Failed to close partially constructed MAGPIE audio writer: {exc}")
            raise
        self._audio_writer = audio_writer
        self._event_writer = event_writer
        self._client_gid: FrameId | None = None
        self._response_key: str | None = None
        self._audio_gid: FrameId | None = None
        self._audio_frame_id = 0
        self._latest_response_key: str | None = None
        self._latest_audio_gid: FrameId | None = None
        self._latest_audio_frame_id = 0
        self._cancelled_response_keys: set[str] = set()
        self._shutdown = False
        self._shutdown_complete = False
        self._audio_writer_closed = False
        self._event_writer_closed = False

    def _write_audio(self, frame: dict[str, Any]) -> None:
        try:
            self._audio_writer.write(frame, self._audio_topic)
        except Exception as exc:
            raise MagpieSinkError(f"MAGPIE audio output write failed: {exc}") from exc

    def _write_event(self, frame: dict[str, Any]) -> None:
        try:
            self._event_writer.write(frame, self._event_topic)
        except Exception as exc:
            raise MagpieSinkError(f"MAGPIE event output write failed: {exc}") from exc

    def open_session(self, client_gid: FrameId) -> None:
        """Correlate every outgoing frame with the active MAGPIE client."""

        if self._shutdown:
            raise RuntimeError("MAGPIE S2S sink is shut down")
        self._client_gid = client_gid

    async def send_events(self, events: list[Any]) -> None:
        if self._shutdown:
            return
        for event in events:
            payload = _event_payload(event)
            event_type = payload.get("type")
            if event_type == "response.output_audio.done":
                self._finish_audio_response()

            self._write_event(DictFrame(gid=self._client_gid, value=payload).to_dict())

    async def _send_session_lifecycle(
        self,
        client_gid: FrameId,
        event_type: str,
    ) -> None:
        """Send a transport lifecycle event on an explicit client GID."""

        if self._shutdown:
            return
        self._write_event(
            DictFrame(
                gid=client_gid,
                value={"type": event_type},
            ).to_dict()
        )

    async def send_session_closing(self, client_gid: FrameId) -> None:
        """Acknowledge that the close request was accepted before draining."""

        await self._send_session_lifecycle(client_gid, "magpie.session.closing")

    async def send_session_closed(self, client_gid: FrameId) -> None:
        """Report that a session has fully drained on the requesting client GID."""

        await self._send_session_lifecycle(client_gid, "magpie.session.closed")

    async def send_audio(
        self,
        pcm: bytes,
        response_key: str | None = None,
    ) -> None:
        if self._shutdown or not pcm:
            return
        if response_key is not None and response_key in self._cancelled_response_keys:
            Logger.debug(f"Discarding late PCM for cancelled response: {response_key}")
            return
        if self._audio_gid is None or (
            response_key is not None and self._response_key != response_key
        ):
            self._finish_audio_response()
            self._audio_gid = get_uinque_id()
            self._response_key = response_key or f"audio_{self._audio_gid}"
            self._audio_frame_id = 0
            self._latest_response_key = self._response_key
            self._latest_audio_gid = self._audio_gid
            self._latest_audio_frame_id = 0

        self._audio_frame_id += 1
        self._latest_audio_frame_id = self._audio_frame_id
        frame = S2SAudioFrame(
            gid=self._audio_gid,
            id=self._audio_frame_id,
            channels=PIPELINE_CHANNELS,
            sample_rate=PIPELINE_SAMPLE_RATE,
            bit_depth=PIPELINE_BIT_DEPTH,
            data=pcm,
            client_gid=self._client_gid,
            response_key=self._response_key,
        )
        self._write_audio(frame.to_dict())

    def discard_pending_audio(self) -> None:
        # Direct MAGPIE output has no local queue to drain.  Send an exact,
        # response-keyed cancellation both in audio-stream order and as a
        # control event.  This remains correct even when the two streams arrive
        # in different orders at the client.
        self._cancel_audio_response()

    def _finish_audio_response(self, *, cancelled: bool = False) -> None:
        if self._audio_gid is None:
            return
        self._audio_frame_id += 1
        response_key = self._response_key or f"audio_{self._audio_gid}"
        frame = S2SAudioFrame(
            gid=self._audio_gid,
            id=self._audio_frame_id,
            channels=PIPELINE_CHANNELS,
            sample_rate=PIPELINE_SAMPLE_RATE,
            bit_depth=PIPELINE_BIT_DEPTH,
            data=b"",
            client_gid=self._client_gid,
            response_key=response_key,
            cancelled=cancelled,
        )
        try:
            self._write_audio(frame.to_dict())
        finally:
            # A failed terminal write makes the session unusable, but must not
            # leave response state that poisons close/shutdown retries.
            self._response_key = None
            self._audio_gid = None
            self._audio_frame_id = 0
        self._latest_response_key = response_key
        self._latest_audio_gid = frame.gid
        self._latest_audio_frame_id = frame.id

    def _cancel_audio_response(self) -> None:
        response_key = self._response_key or self._latest_response_key
        if response_key is None or response_key in self._cancelled_response_keys:
            return

        failure: Exception | None = None
        try:
            if self._audio_gid is not None:
                self._finish_audio_response(cancelled=True)
            elif self._latest_audio_gid is not None:
                # Audio may already have been fully produced while QTrobot still has
                # it buffered.  A second, cancelled terminal for the same response
                # is valid and remains ordered before any subsequently written PCM.
                self._latest_audio_frame_id += 1
                frame = S2SAudioFrame(
                    gid=self._latest_audio_gid,
                    id=self._latest_audio_frame_id,
                    channels=PIPELINE_CHANNELS,
                    sample_rate=PIPELINE_SAMPLE_RATE,
                    bit_depth=PIPELINE_BIT_DEPTH,
                    data=b"",
                    client_gid=self._client_gid,
                    response_key=response_key,
                    cancelled=True,
                )
                self._write_audio(frame.to_dict())
        except Exception as exc:
            failure = exc

        self._cancelled_response_keys.add(response_key)
        try:
            self._write_event(
                DictFrame(
                    gid=self._client_gid,
                    value={
                        "type": "magpie.audio.cancelled",
                        "response_key": response_key,
                    },
                ).to_dict()
            )
        except Exception as exc:
            if failure is None:
                failure = exc
        if failure is not None:
            raise failure
        Logger.debug(f"MAGPIE audio response cancelled: {response_key}")

    def _reset_audio_tracking(self) -> None:
        self._response_key = None
        self._audio_gid = None
        self._audio_frame_id = 0
        self._latest_response_key = None
        self._latest_audio_gid = None
        self._latest_audio_frame_id = 0
        self._cancelled_response_keys.clear()

    async def close(self) -> None:
        """End delivery for one session while keeping bound streams reusable."""

        try:
            self._finish_audio_response()
        finally:
            self._reset_audio_tracking()
            self._client_gid = None

    async def shutdown(self) -> None:
        """Permanently close the MAGPIE publishers during process shutdown."""

        if self._shutdown_complete:
            return
        self._shutdown = True
        failures: list[Exception] = []
        try:
            self._finish_audio_response()
        except Exception as exc:
            failures.append(exc)
        finally:
            self._reset_audio_tracking()
            self._client_gid = None

        # Direct writers are created and used by this event-loop thread; close
        # their ZMQ sockets on the same thread as well. Each resource is tried
        # independently so one broken socket cannot leak the other.
        for label, writer, closed_attr in (
            ("audio", self._audio_writer, "_audio_writer_closed"),
            ("event", self._event_writer, "_event_writer_closed"),
        ):
            if getattr(self, closed_attr):
                continue
            try:
                writer.close()
            except Exception as exc:
                failures.append(MagpieSinkError(f"Failed to close MAGPIE {label} writer: {exc}"))
            else:
                setattr(self, closed_attr, True)

        self._shutdown_complete = self._audio_writer_closed and self._event_writer_closed
        if failures:
            raise MagpieSinkError(f"MAGPIE sink shutdown failed: {failures[0]}") from failures[0]


__all__ = [
    "MagpieSessionSink",
    "MagpieSinkError",
    "StrictZmqStreamReader",
    "StrictZmqStreamWriter",
]
