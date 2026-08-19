"""Native MAGPIE output sink for the S2S session runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from luxai.magpie.frames import DictFrame
from luxai.magpie.transport import ZmqStreamWriter
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
        self._audio_writer = ZmqStreamWriter(
            audio_endpoint,
            queue_size=audio_queue_size,
            bind=True,
            delivery="reliable",
        )
        self._event_writer = ZmqStreamWriter(
            event_endpoint,
            queue_size=event_queue_size,
            bind=True,
            delivery="reliable",
        )
        self._client_gid: FrameId | None = None
        self._response_key: str | None = None
        self._audio_gid: FrameId | None = None
        self._audio_frame_id = 0
        self._latest_response_key: str | None = None
        self._latest_audio_gid: FrameId | None = None
        self._latest_audio_frame_id = 0
        self._cancelled_response_keys: set[str] = set()
        self._shutdown = False

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

            self._event_writer.write(
                DictFrame(gid=self._client_gid, value=payload).to_dict(),
                self._event_topic,
            )

    async def _send_session_lifecycle(
        self,
        client_gid: FrameId,
        event_type: str,
    ) -> None:
        """Send a transport lifecycle event on an explicit client GID."""

        if self._shutdown:
            return
        self._event_writer.write(
            DictFrame(
                gid=client_gid,
                value={"type": event_type},
            ).to_dict(),
            self._event_topic,
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
        self._audio_writer.write(frame.to_dict(), self._audio_topic)

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
        self._audio_writer.write(frame.to_dict(), self._audio_topic)
        self._latest_response_key = response_key
        self._latest_audio_gid = self._audio_gid
        self._latest_audio_frame_id = self._audio_frame_id
        self._response_key = None
        self._audio_gid = None
        self._audio_frame_id = 0

    def _cancel_audio_response(self) -> None:
        response_key = self._response_key or self._latest_response_key
        if response_key is None or response_key in self._cancelled_response_keys:
            return

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
            self._audio_writer.write(frame.to_dict(), self._audio_topic)

        self._cancelled_response_keys.add(response_key)
        self._event_writer.write(
            DictFrame(
                gid=self._client_gid,
                value={
                    "type": "magpie.audio.cancelled",
                    "response_key": response_key,
                }
            ).to_dict(),
            self._event_topic,
        )
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

        self._finish_audio_response()
        self._reset_audio_tracking()
        self._client_gid = None

    async def shutdown(self) -> None:
        """Permanently close the MAGPIE publishers during process shutdown."""

        if self._shutdown:
            return
        self._finish_audio_response()
        self._reset_audio_tracking()
        self._client_gid = None
        self._shutdown = True
        # Direct writers are created and used by this event-loop thread; close
        # their ZMQ sockets on the same thread as well.
        self._audio_writer.close()
        self._event_writer.close()
