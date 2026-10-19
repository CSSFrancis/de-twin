"""A pool of CPU worker threads, one per logical processor, for frame-parallel numba kernels.

numba's own threads split one call's loop between them, so they wait for the slowest at the
end of every call, and Windows leaves them wherever it likes: on this kind of kernel (a whole
camera frame per task, memory-bound in places) two of them often share a core while another
core idles. Here each thread is pinned to its own logical processor and takes the next job
as soon as it finishes one, and the caller submits jobs ahead (a frame each) without waiting.

The threads run below normal priority: the twin feeds a consumer (DE-Server reading frames
out of shared memory), and a consumer that cannot get a core is a camera that drops frames.
``DE_TWIN_FRAME_THREADS`` sets their number (default: every logical processor).
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from typing import Callable, Optional

log = logging.getLogger(__name__)

_WINDOWS = sys.platform == "win32"


def default_threads() -> int:
    n = int(os.environ.get("DE_TWIN_FRAME_THREADS", "0") or 0)
    return max(1, n or os.cpu_count() or 1)


def _pin_and_lower(cpu: Optional[int]) -> None:
    """Pin the calling thread to logical processor *cpu* (None: leave it) and lower its
    priority. Best effort: a refusal leaves the thread as it was."""
    try:
        if _WINDOWS:
            import ctypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            me = k32.GetCurrentThread()
            if cpu is not None and cpu < 64:  # one processor group
                k32.SetThreadAffinityMask(me, 1 << cpu)
            k32.SetThreadPriority(me, -1)  # THREAD_PRIORITY_BELOW_NORMAL
        elif cpu is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {cpu})  # Linux: the calling thread
    except Exception:  # noqa: BLE001 - pinning is an optimisation only
        log.debug("could not pin a worker thread", exc_info=True)


class Batch:
    """Jobs submitted together; :meth:`wait` returns their results in submission order."""

    def __init__(self, count: int):
        self._left = count
        self._results: list = [None] * count
        self._error: Optional[BaseException] = None
        self._done = threading.Event()
        self._lock = threading.Lock()
        if count == 0:
            self._done.set()

    def _finish(self, index: int, result, error: Optional[BaseException]) -> None:
        with self._lock:
            self._results[index] = result
            if error is not None and self._error is None:
                self._error = error
            self._left -= 1
            if self._left == 0:
                self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def wait_done(self, timeout: Optional[float] = None) -> bool:
        """Wait for the jobs, up to *timeout*; whether they are done (errors not raised)."""
        return self._done.wait(timeout)

    def wait(self, timeout: Optional[float] = None) -> list:
        if not self._done.wait(timeout):
            raise TimeoutError("frame jobs did not finish")
        if self._error is not None:
            raise self._error
        return self._results


class Workers:
    """*threads* pinned worker threads (default :func:`default_threads`). A job is
    ``fn(local, *args)``, where ``local`` is a dict private to the worker running it (for its
    scratch buffers)."""

    def __init__(self, threads: Optional[int] = None, *, pin: bool = True):
        self.threads = int(threads or default_threads())
        self._jobs: "queue.SimpleQueue" = queue.SimpleQueue()
        cpus = os.cpu_count() or self.threads
        self._threads = [
            threading.Thread(target=self._run, args=(i % cpus if pin else None,),
                             name=f"de-twin-frames-{i}", daemon=True)
            for i in range(self.threads)
        ]
        for t in self._threads:
            t.start()

    def _run(self, cpu: Optional[int]) -> None:
        _pin_and_lower(cpu)
        local: dict = {}
        while True:
            item = self._jobs.get()
            if item is None:
                return
            fn, args, batch, index = item
            try:
                result, error = fn(local, *args), None
            except BaseException as exc:  # noqa: BLE001 - handed to the waiter
                result, error = None, exc
            batch._finish(index, result, error)
            del item, fn, args, batch, result  # a finished job holds nothing (its frame buffer)

    def submit(self, fn: Callable, arg_list: list) -> Batch:
        """Run ``fn(local, *args)`` for each ``args`` of *arg_list*; returns their `Batch`."""
        batch = Batch(len(arg_list))
        for i, args in enumerate(arg_list):
            self._jobs.put((fn, args, batch, i))
        return batch

    def close(self) -> None:
        for _ in self._threads:
            self._jobs.put(None)


_POOL: Optional[Workers] = None
_POOL_LOCK = threading.Lock()


def pool() -> Workers:
    """The process-wide pool, started on first use."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = Workers()
        return _POOL
