"""Shared-memory face: the twin as DE-Server's external frame source.

DE-Server (test pattern "External Frame Source (Shared Memory)") creates the shared
memory and publishes each acquisition's request. This face turns the request into an
``AcquisitionRequest``, renders frames through the twin and publishes them, paced to the
frame time and throttled by DE-Server's reads.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

from ..transport import shm_layout as L

log = logging.getLogger(__name__)

#: DE-Server gives up on the producer when a frame is ~1 s late (GrabberSim: two frame
#: times + 1000 ms) and stops the acquisition. A render can take longer than that — the
#: first view of a new magnification, the full render after a drag — so when no new
#: frame is ready after this long, the last one is published again.
KEEPALIVE_S = 0.5
_DONE = object()


class ShmFace:
    """Serve DE-Server's requests from ``twin``.

    ``warm_up`` renders one frame before attaching: a fresh twin spends seconds on its
    first frame (imports, GPU start-up, building the specimen), while DE-Server waits about
    one frame time plus a second before it gives up on the acquisition. ``ready`` is set
    once the face is warm and serving.
    """

    def __init__(self, twin, name: str = L.DEFAULT_NAME, pace: bool = True, warm_up: bool = True):
        self.twin = twin
        self.name = name
        self.pace = pace
        self.warm_up = warm_up
        self.ready = threading.Event()
        self.producer = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_published = 0
        self.keepalives = 0
        self.requests_served = 0

    def start(self) -> "ShmFace":
        self._thread = threading.Thread(target=self._run, name="de-twin-shm", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.producer is not None:
            self.producer.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---------------------------------------------------------------- loop
    def _attach(self) -> bool:
        from ..transport.shm import FrameProducer

        try:
            self.producer = FrameProducer(self.name)
            log.info("attached to shared memory %s", self.name)
            return True
        except FileNotFoundError:
            return False

    def _warm_up(self) -> None:
        try:
            req = self.twin.request(frame_time_s=0.01, total_frames=1)
            next(iter(self.twin.frames(req)))
        except Exception:  # a failed warm-up only costs the first request its speed
            log.exception("warm-up render failed")

    def _run(self) -> None:
        if self.warm_up:
            self._warm_up()
        self.ready.set()
        pending = None
        while not self._stop.is_set():
            if self.producer is None:
                if not self._attach():
                    self._stop.wait(0.5)  # DE-Server creates the mapping at its first acquisition
                continue
            got = pending or self.producer.poll_request(timeout=0.05)
            pending = None
            if got is None:
                continue
            self.requests_served += 1
            try:
                pending = self._serve(*got)
            except Exception:  # keep serving later requests
                log.exception("failed serving request %d", got[0])

    def _serve(self, request_id: int, request):
        """Publish frames for one request; returns a newer request if one pre-empts it.

        The twin renders on a thread of its own, so a slow render does not starve
        DE-Server: while it runs, the last frame is republished every `KEEPALIVE_S`.
        """
        p = self.producer
        h = p.hdr
        shape = (h.frame_height, h.frame_width)
        dtype = np.uint8 if h.bytes_per_pixel == 1 else np.uint16
        stop = threading.Event()
        frames: "queue.Queue" = queue.Queue(maxsize=2)

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    frames.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def render() -> None:
            try:
                for item in self.twin.frames(request, pace=self.pace, stop=stop):
                    if not put(item):
                        return
            except Exception as e:  # noqa: BLE001 - raised on the serving thread
                put(e)
                return
            put(_DONE)

        worker = threading.Thread(target=render, name="de-twin-shm-render", daemon=True)
        worker.start()
        last = None
        try:
            while True:
                try:
                    item = frames.get(timeout=KEEPALIVE_S)
                except queue.Empty:
                    if self._stop.is_set() or p.request_changed(request_id):
                        return p.poll_request()
                    if last is None:
                        continue
                    item = last
                    self.keepalives += 1
                if item is _DONE:
                    return None
                if isinstance(item, BaseException):
                    raise item
                raw, meta = item
                while not p.wait_slot_free(timeout=0.05):
                    if self._stop.is_set() or p.request_changed(request_id):
                        return p.poll_request()
                p.publish(_fit(raw, shape, dtype), request_id=request_id,
                          frame_index=meta.frame_index,
                          flags=L.FLAG_BLANKED if meta.blanked else 0)
                self.frames_published += 1
                last = item
                if self._stop.is_set() or p.request_changed(request_id):
                    return p.poll_request()
        finally:
            stop.set()  # the renderer stops at its next frame; the twin's lock orders them

def _fit(frame: np.ndarray, shape: tuple[int, int], dtype) -> np.ndarray:
    """Crop or zero-pad to exactly the hw_frame DE-Server asked for."""
    if frame.shape == shape and frame.dtype == dtype:
        return frame
    out = np.zeros(shape, dtype)
    h, w = min(shape[0], frame.shape[0]), min(shape[1], frame.shape[1])
    out[:h, :w] = np.clip(frame[:h, :w], 0, np.iinfo(dtype).max)
    return out
