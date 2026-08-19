"""Bounded queues and overload accounting for the realtime pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from queue import Queue
from typing import Callable, Generic, TypeVar

from luxai.magpie.utils import Logger

from speech_to_speech.pipeline.control import PipelineControlMessage

ItemT = TypeVar("ItemT")


def _is_pipeline_end(item: object) -> bool:
    # Kept local to avoid importing pipeline.messages (and its OpenAI runtime
    # dependencies) in this small queue primitive.
    return isinstance(item, bytes) and item == b"END"


@dataclass(frozen=True)
class PipelineQueueMetrics:
    """A consistent snapshot of one queue's bounded-runtime counters."""

    name: str
    size: int
    max_size: int
    high_watermark: int
    overloads: int
    dropped_items: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


class BoundedPipelineQueue(Queue[ItemT], Generic[ItemT]):
    """A named bounded queue with control-aware overload handling.

    Normal pipeline data uses :class:`queue.Queue` backpressure. A
    ``PipelineControlMessage`` must never block session teardown behind a full
    data queue, so it atomically evicts the oldest non-control item when
    necessary. Every such overload remains observable through ``metrics()``.
    """

    def __init__(self, *, name: str, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"Pipeline queue {name!r} maxsize must be greater than zero")
        super().__init__(maxsize=maxsize)
        self.name = name
        self._high_watermark = 0
        self._overloads = 0
        self._dropped_items = 0

    def _put(self, item: ItemT) -> None:
        super()._put(item)
        self._high_watermark = max(self._high_watermark, self._qsize())

    def _put_with_oldest_drop(
        self,
        item: ItemT,
        *,
        droppable: Callable[[ItemT], bool],
        drop_oldest_if_none: bool = False,
    ) -> tuple[bool, int, int]:
        """Insert *item* without blocking, evicting one eligible old item.

        Returns ``(accepted, dropped_count, overload_count)``. Queue internals
        are manipulated under the queue's own mutex so a consumer cannot race
        the full-check/eviction/insertion transaction.
        """

        dropped = 0
        with self.not_full:
            if self._qsize() >= self.maxsize:
                self._overloads += 1
                drop_index = next(
                    (index for index, queued in enumerate(self.queue) if droppable(queued)),
                    None,
                )
                if drop_index is None and drop_oldest_if_none:
                    drop_index = 0
                if drop_index is None:
                    # No queued item was eligible, so the incoming item is the
                    # one rejected by this bounded overload policy.
                    self._dropped_items += 1
                    return False, 1, self._overloads
                del self.queue[drop_index]
                # Discarded items will never call task_done(). Preserve Queue's
                # unfinished-task accounting for callers that use join().
                if self.unfinished_tasks > 0:
                    self.unfinished_tasks -= 1
                    if self.unfinished_tasks == 0:
                        self.all_tasks_done.notify_all()
                self._dropped_items += 1
                dropped = 1

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
            return True, dropped, self._overloads

    def put(
        self,
        item: ItemT,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        if isinstance(item, PipelineControlMessage) or _is_pipeline_end(item):
            accepted, dropped, overloads = self._put_with_oldest_drop(
                item,
                droppable=lambda queued: not isinstance(queued, PipelineControlMessage)
                and not _is_pipeline_end(queued),
                # An impossible-under-normal-operation queue made entirely of
                # controls/termination sentinels must still stay bounded and
                # must not deadlock shutdown. Keep the newest terminal item.
                drop_oldest_if_none=True,
            )
            if accepted:
                if dropped:
                    self._log_overload(overloads)
                return
        super().put(item, block=block, timeout=timeout)

    def metrics(self) -> PipelineQueueMetrics:
        with self.mutex:
            return PipelineQueueMetrics(
                name=self.name,
                size=self._qsize(),
                max_size=self.maxsize,
                high_watermark=self._high_watermark,
                overloads=self._overloads,
                dropped_items=self._dropped_items,
            )

    def _log_overload(self, overloads: int) -> None:
        # Log the first overload and then powers of two. This remains visible
        # under sustained overload without flooding a robot's journal.
        if overloads > 0 and (overloads == 1 or overloads & (overloads - 1) == 0):
            metrics = self.metrics()
            Logger.warning(
                f"Pipeline queue {self.name} overloaded {metrics.overloads} time(s); "
                f"dropped={metrics.dropped_items}, size={metrics.size}/{metrics.max_size}"
            )


class AudioIngressQueue(BoundedPipelineQueue[ItemT], Generic[ItemT]):
    """Latest-wins queue for realtime PCM input.

    Audio older than the bounded latency window is no longer useful to a live
    conversation. When full, replace the oldest audio tuple instead of blocking
    the asyncio transport loop or allowing latency and memory use to grow.
    Control messages retain the priority behavior of ``BoundedPipelineQueue``.
    """

    @staticmethod
    def _is_audio_item(item: object) -> bool:
        # VAD input is ``(pcm_bytes, runtime_config)``. Pipeline controls are
        # dataclasses and therefore cannot be mistaken for realtime audio.
        return isinstance(item, tuple)

    def put(
        self,
        item: ItemT,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        if self._is_audio_item(item):
            _accepted, dropped, overloads = self._put_with_oldest_drop(
                item,
                droppable=self._is_audio_item,
            )
            if dropped:
                self._log_overload(overloads)
            return
        super().put(item, block=block, timeout=timeout)


__all__ = [
    "AudioIngressQueue",
    "BoundedPipelineQueue",
    "PipelineQueueMetrics",
]
