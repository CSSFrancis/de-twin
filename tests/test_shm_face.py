"""The shared-memory face serving a Python FrameConsumer (standing in for DE-Server)."""

import os
import uuid

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.faces.shm_face import ShmFace
from de_twin.state import ExposureMode
from de_twin.transport import shm_layout as L
from de_twin.transport.shm import FrameConsumer
from de_twin.twin import DigitalTwin


@pytest.fixture
def served():
    name = f"DE_ExternalFramesFace_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    consumer = FrameConsumer(name, max_frame_bytes=1024 * 1024 * 2)  # DE-Server creates the mapping
    face = ShmFace(twin, name=name, pace=False).start()
    yield twin, face, consumer
    face.stop()
    consumer.close()


def test_frames_arrive_in_order(served):
    twin, face, consumer = served
    consumer.begin(frame_shape=(1024, 1024), frame_time_s=0.01, total_frames=10)
    frames = [consumer.read(timeout=10) for _ in range(10)]
    assert [h["frame_index"] for _, h in frames] == list(range(10))
    assert all(f.shape == (1024, 1024) and f.dtype == np.uint16 for f, _ in frames)
    assert frames[0][0].mean() > 450  # beam on
    assert not frames[0][1]["flags"] & L.FLAG_BLANKED


def test_dark_request_produces_blanked_frames(served):
    twin, face, consumer = served
    consumer.begin(frame_shape=(1024, 1024), total_frames=3, exposure_mode=ExposureMode.DARK)
    for _ in range(3):
        frame, info = consumer.read(timeout=10)
        assert info["flags"] & L.FLAG_BLANKED
        assert 300 < frame.mean() < 450


def test_new_request_preempts_live_stream(served):
    twin, face, consumer = served
    consumer.begin(frame_shape=(1024, 1024), total_frames=0)  # live: until told otherwise
    for _ in range(3):
        consumer.read(timeout=10)
    rid = consumer.begin(frame_shape=(512, 512), total_frames=2, binning=(2, 2), sensor_shape=(1024, 1024))
    frame, info = consumer.read(timeout=10)
    assert info["request_id"] == rid and info["frame_index"] == 0
    assert frame.shape == (512, 512)
