import threading
import time
from collections.abc import Sequence
from typing import Any

from luxai.magpie.utils import Logger


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers: Sequence[Any]) -> None:
        self.handlers = handlers
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run)
            # Graceful shutdown still joins handlers. Daemon mode only prevents
            # one wedged third-party backend from trapping the whole service.
            thread.daemon = True
            self.threads.append(thread)
            thread.start()

    def wait(self) -> None:
        for thread in self.threads:
            thread.join()

    def stop(self, timeout: float = 5.0) -> None:
        # Signal all handlers to stop
        for handler in self.handlers:
            handler.stop_event.set()

        # Bound the whole shutdown, rather than spending the timeout once per
        # handler. Surviving daemon handlers cannot hold process exit hostage.
        deadline = time.monotonic() + max(0.0, float(timeout))
        for i, thread in enumerate(self.threads):
            if thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    Logger.warning(
                        f"Thread {i} ({thread.name}) did not terminate within "
                        f"{timeout:g} seconds"
                    )
