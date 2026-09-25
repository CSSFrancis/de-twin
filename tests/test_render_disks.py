"""Large disks drawn over the detector only are the stamped disks."""

import numpy as np
import pytest

from de_twin.render import diffraction as D
from de_twin.render import disks

pytestmark = pytest.mark.skipif(not disks.AVAILABLE, reason="numba missing")


@pytest.mark.parametrize("n,radius", [(48, 550.0), (200, 40.0), (60, 24.0)])
def test_drawn_disks_are_the_stamped_ones(monkeypatch, n, radius):
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-300, 556, n), rng.uniform(-300, 556, n)
    w = rng.uniform(0, 1, n)
    sigma = D.blur_sigma_px(radius)
    stamped = np.zeros((256, 256), np.float32)
    drawn = np.zeros_like(stamped)
    monkeypatch.setattr(D, "DRAW_DISKS_ABOVE_PX", 1e9)
    D.stamp_disks(stamped, x, y, w, radius, sigma)
    monkeypatch.setattr(D, "DRAW_DISKS_ABOVE_PX", 0.0)
    D.stamp_disks(drawn, x, y, w, radius, sigma)
    np.testing.assert_allclose(drawn, stamped, atol=1e-4 * stamped.max())


def test_a_whole_disk_integrates_to_its_weight():
    out = np.zeros((512, 512), np.float32)
    disks.draw_disks(out, [256.3], [255.8], [2.5], 100.0, 4.0)
    assert out.sum() == pytest.approx(2.5, rel=1e-4)
