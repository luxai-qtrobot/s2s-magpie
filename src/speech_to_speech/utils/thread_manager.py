from __future__ import annotations

import math
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from luxai.magpie.utils import Logger


@dataclass(frozen=True)
class HandlerThreadFailure:
    """Fatal health state reported by one pipeline handler thread."""

    index: int
    handler_name: str
    thread_name: str
    reason: str
    traceback_text: str | None = None

    def describe(self) -> str:
        return f"handler {self.index} ({self.handler_name}, {self.thread_name}) {self.reason}"


class HandlerThreadError(RuntimeError):
    def __init__(self, failure: HandlerThreadFailure) -> None:
        super().__init__(failure.describe())
        self.failure = failure


class ThreadManager:
    """Start, supervise, and stop the serial pipeline handler threads.

    Handler threads are daemonized because Python cannot interrupt a backend
    blocked inside native CUDA/provider code. Unexpected exit and lack of
    progress are converted into a fatal health result; the service loop then
    exits nonzero so an external supervisor can replace the process.
    """

    def __init__(
        self,
        handlers: Sequence[Any],
        *,
        stall_timeout: float | None = 180.0,
    ) -> None:
        if stall_timeout is not None and (not math.isfinite(stall_timeout) or stall_timeout <= 0):
            raise ValueError("stall_timeout must be a finite value greater than zero or None")
        self.handlers = handlers
        self.stall_timeout = stall_timeout
        self.threads: list[threading.Thread] = []
        self._failure: HandlerThreadFailure | None = None
        self._failure_lock = threading.Lock()
        self._stopping = threading.Event()
        self._started = False

    def _shutdown_expected(self, handler: Any) -> bool:
        handler_stop = getattr(handler, "stop_event", None)
        return self._stopping.is_set() or bool(
            handler_stop is not None
            and callable(getattr(handler_stop, "is_set", None))
            and handler_stop.is_set()
        )

    def _record_failure(self, failure: HandlerThreadFailure) -> HandlerThreadFailure:
        with self._failure_lock:
            if self._failure is None:
                self._failure = failure
                Logger.error(f"Fatal S2S pipeline worker failure: {failure.describe()}")
                if failure.traceback_text:
                    Logger.error(failure.traceback_text.rstrip())
            return self._failure

    def _run_handler(self, index: int, handler: Any) -> None:
        thread_name = threading.current_thread().name
        handler_name = type(handler).__name__
        try:
            handler.run()
        except BaseException as exc:
            if not self._shutdown_expected(handler):
                self._record_failure(
                    HandlerThreadFailure(
                        index=index,
                        handler_name=handler_name,
                        thread_name=thread_name,
                        reason=f"raised {type(exc).__name__}: {exc}",
                        traceback_text=traceback.format_exc(),
                    )
                )
            return

        if not self._shutdown_expected(handler):
            self._record_failure(
                HandlerThreadFailure(
                    index=index,
                    handler_name=handler_name,
                    thread_name=thread_name,
                    reason="exited unexpectedly",
                )
            )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("ThreadManager can only be started once")
        self._started = True
        for index, handler in enumerate(self.handlers):
            thread = threading.Thread(
                target=self._run_handler,
                args=(index, handler),
                name=f"s2s-{index}-{type(handler).__name__}",
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()

    def check_health(self) -> HandlerThreadFailure | None:
        """Return the first fatal exit/stall observed, if any."""

        with self._failure_lock:
            failure = self._failure
        if failure is not None or self._stopping.is_set():
            return failure

        for index, (handler, thread) in enumerate(zip(self.handlers, self.threads)):
            if not thread.is_alive() and not self._shutdown_expected(handler):
                return self._record_failure(
                    HandlerThreadFailure(
                        index=index,
                        handler_name=type(handler).__name__,
                        thread_name=thread.name,
                        reason="is no longer alive",
                    )
                )

            if self.stall_timeout is None:
                continue
            snapshot = getattr(handler, "worker_activity_snapshot", None)
            if not callable(snapshot):
                continue
            try:
                processing, stalled_for, item_label = snapshot()
            except Exception as exc:
                return self._record_failure(
                    HandlerThreadFailure(
                        index=index,
                        handler_name=type(handler).__name__,
                        thread_name=thread.name,
                        reason=f"health probe failed: {type(exc).__name__}: {exc}",
                    )
                )
            if processing and stalled_for >= self.stall_timeout:
                return self._record_failure(
                    HandlerThreadFailure(
                        index=index,
                        handler_name=type(handler).__name__,
                        thread_name=thread.name,
                        reason=(
                            f"made no progress for {stalled_for:.1f}s while processing "
                            f"{item_label or 'an item'}"
                        ),
                    )
                )
        return None

    def wait(self) -> None:
        for thread in self.threads:
            thread.join()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        for handler in self.handlers:
            handler.stop_event.set()

        # Bound the whole shutdown, rather than spending the timeout once per
        # handler. Surviving daemon handlers cannot hold process exit hostage.
        deadline = time.monotonic() + max(0.0, float(timeout))
        for index, thread in enumerate(self.threads):
            if thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    Logger.warning(
                        f"Thread {index} ({thread.name}) did not terminate within "
                        f"{timeout:g} seconds"
                    )


__all__ = [
    "HandlerThreadError",
    "HandlerThreadFailure",
    "ThreadManager",
]
