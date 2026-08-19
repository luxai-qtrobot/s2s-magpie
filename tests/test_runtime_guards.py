from __future__ import annotations

import sys
import time
import types
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from queue import Empty, Full
from threading import Event

import pytest

if find_spec("openai") is None:
    # Import the focused queue modules without triggering the pipeline package's
    # optional OpenAI-facing re-exports in a lightweight development checkout.
    pipeline_module = types.ModuleType("speech_to_speech.pipeline")
    pipeline_module.__path__ = [
        str(Path(__file__).parents[1] / "src" / "speech_to_speech" / "pipeline")
    ]
    sys.modules.setdefault("speech_to_speech.pipeline", pipeline_module)

if find_spec("luxai.magpie") is None:
    # Keep these focused tests runnable in a lightweight checkout. Production
    # environments import the real Logger from the required luxai-magpie wheel.
    class _Logger:
        @staticmethod
        def error(_message: str) -> None:
            pass

        @staticmethod
        def warning(_message: str) -> None:
            pass

    luxai_module = sys.modules.setdefault("luxai", types.ModuleType("luxai"))
    magpie_module = types.ModuleType("luxai.magpie")
    utils_module = types.ModuleType("luxai.magpie.utils")
    utils_module.Logger = _Logger
    setattr(luxai_module, "magpie", magpie_module)
    sys.modules.setdefault("luxai.magpie", magpie_module)
    sys.modules.setdefault("luxai.magpie.utils", utils_module)

control_module = import_module("speech_to_speech.pipeline.control")
queues_module = import_module("speech_to_speech.pipeline.queues")
PipelineControlMessage = control_module.PipelineControlMessage
SESSION_END = control_module.SESSION_END
AudioIngressQueue = queues_module.AudioIngressQueue
BoundedPipelineQueue = queues_module.BoundedPipelineQueue


def test_audio_ingress_overload_keeps_latest_pcm() -> None:
    queue = AudioIngressQueue(name="audio_input", maxsize=2)

    queue.put((b"one", None))
    queue.put((b"two", None))
    queue.put((b"three", None))

    assert queue.get_nowait() == (b"two", None)
    assert queue.get_nowait() == (b"three", None)
    with pytest.raises(Empty):
        queue.get_nowait()
    metrics = queue.metrics()
    assert metrics.max_size == 2
    assert metrics.high_watermark == 2
    assert metrics.overloads == 1
    assert metrics.dropped_items == 1


def test_session_control_evicts_data_instead_of_blocking() -> None:
    queue = BoundedPipelineQueue(name="stage", maxsize=2)
    queue.put("one")
    queue.put("two")

    control = PipelineControlMessage(SESSION_END.kind, session_id="session-1")
    queue.put(control)

    assert queue.get_nowait() == "two"
    assert queue.get_nowait() == control
    metrics = queue.metrics()
    assert metrics.overloads == 1
    assert metrics.dropped_items == 1


def test_control_only_overload_stays_bounded_and_keeps_latest() -> None:
    queue = BoundedPipelineQueue(name="stage", maxsize=1)
    old_control = PipelineControlMessage(SESSION_END.kind, session_id="old")
    new_control = PipelineControlMessage(SESSION_END.kind, session_id="new")
    queue.put(old_control)

    queue.put(new_control)

    assert queue.qsize() == 1
    assert queue.get_nowait() == new_control
    assert queue.metrics().dropped_items == 1


def test_pipeline_end_evicts_data_instead_of_blocking() -> None:
    queue = BoundedPipelineQueue(name="stage", maxsize=1)
    queue.put("pending-data")

    queue.put(b"END")

    assert queue.qsize() == 1
    assert queue.get_nowait() == b"END"
    assert queue.metrics().dropped_items == 1


def test_normal_pipeline_data_applies_backpressure() -> None:
    queue = BoundedPipelineQueue(name="stage", maxsize=1)
    queue.put_nowait("one")

    with pytest.raises(Full):
        queue.put_nowait("two")


ThreadManager = import_module("speech_to_speech.utils.thread_manager").ThreadManager


class _CrashingHandler:
    def __init__(self) -> None:
        self.stop_event = Event()

    def run(self) -> None:
        raise RuntimeError("worker failed")


class _StalledHandler:
    def __init__(self) -> None:
        self.stop_event = Event()

    def run(self) -> None:
        self.stop_event.wait()

    def worker_activity_snapshot(self) -> tuple[bool, float, str]:
        return True, 10.0, "request"


def test_thread_manager_reports_unexpected_handler_exit() -> None:
    manager = ThreadManager([_CrashingHandler()], stall_timeout=1.0)
    manager.start()

    deadline = time.monotonic() + 1.0
    failure = None
    while failure is None and time.monotonic() < deadline:
        failure = manager.check_health()
        time.sleep(0.005)

    assert failure is not None
    assert "raised RuntimeError: worker failed" in failure.reason
    manager.stop(timeout=0.1)


def test_thread_manager_reports_handler_stall() -> None:
    handler = _StalledHandler()
    manager = ThreadManager([handler], stall_timeout=1.0)
    manager.start()

    failure = manager.check_health()
    assert failure is not None
    assert "made no progress" in failure.reason
    manager.stop(timeout=0.1)
