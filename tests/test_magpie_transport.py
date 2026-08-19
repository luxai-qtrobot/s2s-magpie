from __future__ import annotations

import uuid

import pytest


pytest.importorskip("luxai.magpie")

from luxai.s2s_magpie.transport import (  # noqa: E402
    StrictZmqStreamReader,
    StrictZmqStreamWriter,
)


def test_strict_writer_propagates_serialization_failure() -> None:
    class FailingSerializer:
        def serialize(self, _data: object) -> bytes:
            raise OSError("serialize failed")

    endpoint = f"inproc://strict-writer-{uuid.uuid4()}"
    writer = StrictZmqStreamWriter(
        endpoint,
        serializer=FailingSerializer(),
        queue_size=0,
        bind=True,
    )
    try:
        with pytest.raises(OSError, match="serialize failed"):
            writer.write({"value": 1}, "test")
    finally:
        writer.close()


def test_strict_reader_surfaces_background_transport_failure() -> None:
    class FailingReader(StrictZmqStreamReader):
        def _transport_read_blocking(self, timeout: float | None = None):
            raise OSError("read failed")

    endpoint = f"inproc://strict-reader-{uuid.uuid4()}"
    reader = FailingReader(endpoint, queue_size=1, bind=True)
    try:
        reader.thread.join(timeout=2.0)
        assert not reader.thread.is_alive()
        with pytest.raises(RuntimeError, match="transport failed"):
            reader.read(timeout=0.1)
    finally:
        reader.close()

