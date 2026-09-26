"""SerialEM calibration files to and from a column definition: an exact round trip."""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.optics import OpticsConfig
from de_twin.optics.realism import ColumnRealism
from de_twin.optics.serialem import ColumnDefinition, from_serialem, to_serialem, twin_mag_table
from de_twin.twin import DigitalTwin

CAMERA = "DESim"


@pytest.fixture
def files(tmp_path):
    return tmp_path / "SerialEMcalibrations.txt", tmp_path / "rotation_and_pixel.txt"


def test_the_file_is_a_serialem_calibration_file(files):
    cal, props = files
    to_serialem(ColumnRealism(seed=3), CAMERA, cal, props)
    text = cal.read_text().splitlines()
    assert text[0] == "SerialEMCalibrations"
    keys = {line.split()[0] for line in text[1:] if line and line.split()[0].isalpha()}
    assert {"ImageShiftMatrix", "StageToCameraMatrix", "ImageShiftOffsets", "BeamShiftCalibration",
            "CrossoverIntensity", "HighFocusMagCal", "FocusCalibration"} <= keys
    n = int(next(line for line in text if line.startswith("ImageShiftMatrix")).split()[1])
    assert n == len(twin_mag_table())
    assert all(line.startswith("RotationAndPixel") for line in props.read_text().splitlines()[1:])


@pytest.mark.parametrize("with_props", [False, True])
def test_a_column_survives_the_round_trip(files, with_props):
    cal, props = files
    real = ColumnRealism(seed=11)
    to_serialem(real, CAMERA, cal, props)
    d = from_serialem(cal, props if with_props else None, camera=CAMERA)
    assert isinstance(d, ColumnDefinition)
    for mag in twin_mag_table().values():
        mode = "LowMAG" if mag <= 200 else "MAG1"
        dr = (d.rotation_rad(mode, mag) - real.rotation_rad(mode, mag) + math.pi) % (2 * math.pi) - math.pi
        assert dr == pytest.approx(0.0, abs=1e-6), mag
        assert d.pixel_scale(mode, mag) == pytest.approx(real.pixel_scale(mode, mag), rel=1e-6), mag
        np.testing.assert_allclose(d.is_matrix(mode, mag), real.is_matrix(mode, mag), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(d.mag_offset_um(mode, mag), real.mag_offset_um(mode, mag), atol=1e-6)
    np.testing.assert_allclose(d.bs_matrix(), real.bs_matrix(), rtol=1e-6, atol=1e-7)
    for spot in (1, 3, 5):
        for probe in (0, 1):
            assert d.crossover(spot, probe) == pytest.approx(real.crossover(spot, probe), abs=1e-7)
    assert d.hd_scale_per_um == pytest.approx(real.hd_scale_per_um, rel=1e-5)
    assert d.hd_rotation_deg_per_um == pytest.approx(real.hd_rotation_deg_per_um, rel=1e-5)


def test_a_twin_defined_by_serialem_calibrations_is_that_column(files):
    cal, props = files
    real = ColumnRealism(seed=11, backlash_um=0.0, is_coma_mrad_per_um=0.0, is_astig_nm_per_um=0.0)
    to_serialem(real, CAMERA, cal, props)
    d = from_serialem(cal, props, camera=CAMERA)
    a = DigitalTwin("Dense Au on holey C", camera=CAMERA, clock=ManualClock(), seed=1,
                    optics_config=OpticsConfig(realism=real))
    b = DigitalTwin("Dense Au on holey C", camera=CAMERA, clock=ManualClock(), seed=1,
                    optics_config=OpticsConfig(realism=d))
    for mag in (150.0, 5000.0, 20000.0, 60000.0):
        for tw in (a, b):
            tw.column.set("Magnification", mag)
            tw.column.set("ImageShift", (0.3, -0.2))
        ta, tb = a.calibration_truth(), b.calibration_truth()
        assert tb["true_pixel_nm"] == pytest.approx(ta["true_pixel_nm"], rel=1e-6)
        assert (tb["image_rotation_deg"] - ta["image_rotation_deg"] + 180) % 360 - 180 == pytest.approx(0, abs=1e-4)
        np.testing.assert_allclose(tb["is_matrix_um_per_unit"], ta["is_matrix_um_per_unit"], atol=1e-6)
        np.testing.assert_allclose(tb["mag_offset_um"], ta["mag_offset_um"], atol=1e-6)
        oa, ob = a.optics(a.request()), b.optics(b.request())
        np.testing.assert_allclose(ob.view.center_um, oa.view.center_um, atol=1e-6)
    ia = a.flux(a.request())
    ib = b.flux(b.request())
    assert np.corrcoef(ia.ravel(), ib.ravel())[0, 1] > 0.9999, "the same image"
