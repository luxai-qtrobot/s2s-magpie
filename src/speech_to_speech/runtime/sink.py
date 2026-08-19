"""Transport-independent output boundary for one speech-to-speech session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent


@runtime_checkable
class SessionSink(Protocol):
    """Destination for events and raw pipeline-rate PCM produced by a session.

    A sink owns only delivery. Response lifecycle bookkeeping remains in
    :class:`SessionRuntime`, so transports never need a reference to the
    realtime service or its mutable connection state.
    """

    async def send_events(self, events: list[ServerEvent]) -> None:
        """Deliver ordered server events to the attached client."""

    async def send_audio(self, pcm: bytes, response_key: str | None = None) -> None:
        """Deliver mono PCM16 audio at the pipeline sample rate."""

    def discard_pending_audio(self) -> None:
        """Drop audio buffered by this sink but not yet played by the client."""

    async def close(self) -> None:
        """Close the delivery side of the session."""
