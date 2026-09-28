"""The numba rasteriser (specimen/raster_nb.py) against the NumPy reference: the same pixels."""

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.specimen import raster_nb
from de_twin.twin import DigitalTwin

SPECIMENS = [
    "Dense Au on holey C",
    "Ted Pella 607 - 2160 l/mm grating replica (waffle)",
    "Ted Pella 603 - 2160 l/mm grating replica with 0.261 um latex spheres",
    "Ted Pella 677 - 500 nm cross-line grating replica",
    "Ted Pella 673 - 500 nm cross-line grating replica with 261 nm latex spheres",
    "Ted Pella 628-B - Gold-shadowed latex",
    "Ted Pella 619 - Diffraction standard, evaporated aluminum",
    "Apoferritin in ice",
    "Negative stain on carbon",
    "Au thin film 20 nm",
    "Sparse Au on holey C",
    "Au clusters on lacey C",
]


def _raster(tw, view, fast, monkeypatch):
    monkeypatch.setattr(raster_nb, "AVAILABLE", fast)
    return tw.specimen.rasterize(view)


@pytest.mark.skipif(not raster_nb.AVAILABLE, reason="no numba")
@pytest.mark.parametrize("name", SPECIMENS)
@pytest.mark.parametrize("mag", [3000.0, 25000.0, 150000.0])
def test_the_numba_raster_is_the_numpy_raster(name, mag, monkeypatch):
    import dataclasses

    tw = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=4)
    tw.column.set("Magnification", mag)
    view = dataclasses.replace(tw.optics(tw.request()).view, shape=(384, 448))
    ref = _raster(tw, view, False, monkeypatch)
    new = _raster(tw, view, True, monkeypatch)
    np.testing.assert_array_equal(new.material_id, ref.material_id)
    np.testing.assert_array_equal(new.grain_id, ref.grain_id)
    np.testing.assert_allclose(new.thickness_nm, ref.thickness_nm, rtol=1e-6, atol=1e-5)
    assert (new.under_thickness_nm is None) == (ref.under_thickness_nm is None)
    if ref.under_thickness_nm is not None:
        np.testing.assert_array_equal(new.under_material, ref.under_material)
        np.testing.assert_allclose(new.under_thickness_nm, ref.under_thickness_nm, rtol=1e-6, atol=1e-5)


@pytest.mark.skipif(not raster_nb.AVAILABLE, reason="no numba")
def test_a_rotated_tilted_view(monkeypatch):
    import dataclasses

    tw = DigitalTwin("Ted Pella 603 - 2160 l/mm grating replica with 0.261 um latex spheres", camera="DESim",
                     clock=ManualClock(), seed=4)
    view = dataclasses.replace(tw.optics(tw.request()).view, shape=(300, 340), rotation_rad=0.6, flip_y=True,
                               cos_alpha=0.93)
    ref = _raster(tw, view, False, monkeypatch)
    new = _raster(tw, view, True, monkeypatch)
    np.testing.assert_array_equal(new.material_id, ref.material_id)
    np.testing.assert_array_equal(new.grain_id, ref.grain_id)
    np.testing.assert_allclose(new.thickness_nm, ref.thickness_nm, rtol=1e-6, atol=1e-5)
