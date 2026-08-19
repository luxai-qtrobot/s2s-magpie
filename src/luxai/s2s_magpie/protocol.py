"""MAGPIE protocol shared by the S2S service and its clients.

The service follows the LuxAI driver convention: one configured ZMQ base port
owns the RPC endpoint and every stream endpoint is derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from luxai.magpie.frames import AudioFrameRaw


RPC_PORT_OFFSET = 0
AUDIO_INPUT_PORT_OFFSET = 1
AUDIO_OUTPUT_PORT_OFFSET = 2
EVENT_INPUT_PORT_OFFSET = 3
EVENT_OUTPUT_PORT_OFFSET = 4

STATUS_RPC = "/s2s/status"

AUDIO_INPUT_TOPIC = "/s2s/audio/input"
AUDIO_OUTPUT_TOPIC = "/s2s/audio/output"
EVENT_INPUT_TOPIC = "/s2s/events/input"
EVENT_OUTPUT_TOPIC = "/s2s/events/output"

PIPELINE_SAMPLE_RATE = 16_000
PIPELINE_CHANNELS = 1
PIPELINE_BIT_DEPTH = 16

FrameId = str | int


def service_port(base_port: int, offset: int) -> int:
    """Return and validate a port derived from the service base port."""

    port = int(base_port) + offset
    if not 1 <= port <= 65_535:
        raise ValueError(f"Invalid S2S ZMQ port: {port}")
    return port


def bind_endpoint(base_port: int, offset: int = RPC_PORT_OFFSET) -> str:
    """Return the server-side ZMQ endpoint for a derived service port."""

    return f"tcp://*:{service_port(base_port, offset)}"


def build_system_descriptor(
    node_id: str,
    base_port: int,
    *,
    audio_input_queue_size: int,
    audio_output_queue_size: int,
    event_input_queue_size: int,
    event_output_queue_size: int,
) -> dict:
    """Build the standard MAGPIE discovery descriptor for this service."""

    rpc_endpoint = bind_endpoint(base_port, RPC_PORT_OFFSET)
    return {
        "node_id": str(node_id),
        "rpc": {
            STATUS_RPC: {
                "description": "Get the S2S service and active-session status",
                "params": {},
                "returns": {"type": "dict"},
                "transports": {"zmq": {"endpoint": rpc_endpoint}},
            }
        },
        "stream": {
            AUDIO_INPUT_TOPIC: {
                "direction": "in",
                "frame_type": "AudioFrameRaw",
                "transports": {
                    "zmq": {
                        "endpoint": bind_endpoint(base_port, AUDIO_INPUT_PORT_OFFSET),
                        "delivery": "reliable",
                        "queue_size": int(audio_input_queue_size),
                    }
                },
            },
            AUDIO_OUTPUT_TOPIC: {
                "direction": "out",
                "frame_type": "S2SAudioFrame",
                "transports": {
                    "zmq": {
                        "endpoint": bind_endpoint(base_port, AUDIO_OUTPUT_PORT_OFFSET),
                        "delivery": "reliable",
                        "queue_size": int(audio_output_queue_size),
                    }
                },
            },
            EVENT_INPUT_TOPIC: {
                "direction": "in",
                "frame_type": "DictFrame",
                "transports": {
                    "zmq": {
                        "endpoint": bind_endpoint(base_port, EVENT_INPUT_PORT_OFFSET),
                        "delivery": "reliable",
                        "queue_size": int(event_input_queue_size),
                    }
                },
            },
            EVENT_OUTPUT_TOPIC: {
                "direction": "out",
                "frame_type": "DictFrame",
                "transports": {
                    "zmq": {
                        "endpoint": bind_endpoint(base_port, EVENT_OUTPUT_PORT_OFFSET),
                        "delivery": "reliable",
                        "queue_size": int(event_output_queue_size),
                    }
                },
            },
        },
    }


@dataclass
class S2SAudioFrame(AudioFrameRaw):
    """Raw assistant PCM correlated with one internal S2S response.

    ``client_gid`` identifies the active MAGPIE session and ``response_key``
    remains stable across all PCM chunks and terminal frames for a response.
    A cancelled terminal is carried on the audio stream itself so it cannot be
    reordered behind audio from a newer response; the matching control event
    provides the same correlation to event-only consumers.
    """

    client_gid: FrameId | None = None
    response_key: str = ""
    cancelled: bool = False


__all__ = [
    "AUDIO_INPUT_PORT_OFFSET",
    "AUDIO_INPUT_TOPIC",
    "AUDIO_OUTPUT_PORT_OFFSET",
    "AUDIO_OUTPUT_TOPIC",
    "EVENT_INPUT_PORT_OFFSET",
    "EVENT_INPUT_TOPIC",
    "EVENT_OUTPUT_PORT_OFFSET",
    "EVENT_OUTPUT_TOPIC",
    "FrameId",
    "PIPELINE_BIT_DEPTH",
    "PIPELINE_CHANNELS",
    "PIPELINE_SAMPLE_RATE",
    "RPC_PORT_OFFSET",
    "STATUS_RPC",
    "S2SAudioFrame",
    "bind_endpoint",
    "build_system_descriptor",
    "service_port",
]
