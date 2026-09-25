"""Counting cameras through the numba kernels, and Apollo super-resolution."""

import numpy as np
import pytest

from de_twin.detector import fast
from de_twin.detector.detector import Detector
from de_twin.detector.models import CAMERAS
from de_twin.state import AcquisitionRequest

pytestmark = pytest.mark.skipif(not fast.AVAILABLE, reason="numba missing")

SHAPE = (512, 512)


def _flux(level=5.0):
    f = np.full(SHAPE, level, np.float32)
    f[:, : SHAPE[1] // 2] *= 3.0
    return f


def _det():
    return Detector(CAMERAS["Apollo"], seed=4)


def _req(**kw):
    from de_twin.state import Roi

    return AcquisitionRequest(camera_model="Apollo", hw_roi=Roi(0, 0, *SHAPE[::-1]), **kw)


def test_fast_counting_matches_the_numpy_statistics(monkeypatch):
    flux = _flux(40.0)
    fast_frames = [_det().expose(flux, 0.02, _req(), k)[0] for k in range(4)]
    monkeypatch.setenv("DE_TWIN_NUMPY_DETECTOR", "1")
    ref_frames = [_det().expose(flux, 0.02, _req(), k)[0] for k in range(4)]
    a = np.stack(fast_frames).astype(np.float64)
    b = np.stack(ref_frames).astype(np.float64)
    assert a.dtype == b.dtype and fast_frames[0].dtype == np.uint8
    for sl in (np.s_[:, :, :256], np.s_[:, :, 256:]):  # the bright and the dim half
        assert a[sl].mean() == pytest.approx(b[sl].mean(), rel=0.02)
        assert a[sl].var() == pytest.approx(b[sl].var(), rel=0.05)


def test_frames_differ_and_a_seed_repeats_them():
    d = _det()
    a, _ = d.expose(_flux(), 0.02, _req(), 0)
    b, _ = d.expose(_flux(), 0.02, _req(), 1)
    c, _ = d.expose(_flux(), 0.02, _req(), 0)
    assert not np.array_equal(a, b) and np.array_equal(a, c)


def test_super_resolution_doubles_the_frame_and_keeps_the_counts():
    d = _det()
    flux = _flux()
    std, _ = d.expose(flux, 0.02, _req(), 3)
    sr, info = d.expose(flux, 0.02, _req(super_resolution=True), 3)
    assert info["super_resolution"] and sr.shape == (2 * SHAPE[0], 2 * SHAPE[1]) and sr.dtype == np.uint8
    assert d.output_shape(_req(super_resolution=True)) == sr.shape
    # the same draw: every pixel's counts are spread over its 2x2 sub-pixels
    binned = sr.reshape(SHAPE[0], 2, SHAPE[1], 2).sum(axis=(1, 3))
    np.testing.assert_array_equal(binned, std)
    # sub-pixels are equally likely
    quads = [sr[i::2, j::2].sum() for i in (0, 1) for j in (0, 1)]
    assert max(quads) / min(quads) < 1.05


def test_super_resolution_needs_an_unbinned_counting_camera():
    d = _det()
    assert not d.super_resolution(_req(super_resolution=True, hw_binning=(2, 2)))
    assert not Detector(CAMERAS["DE16"]).super_resolution(AcquisitionRequest(super_resolution=True))
