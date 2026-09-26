"""Ted Pella calibration standards: every product is a preset, and each shows its published
numbers (grating period, sphere size) and the physics of a shadowed replica."""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.specimen import CALIBRATION_STANDARDS, from_name
from de_twin.specimen.materials import MaterialId
from de_twin.specimen.standards import (PRODUCTS, ShadowedReplicaStructure, WaffleGrating, build_structure,
                                        island_film, preset_name)
from de_twin.twin import DigitalTwin


def _twin(number, mag):
    tw = DigitalTwin(preset_name(number), camera="DESim", clock=ManualClock(), seed=1)
    tw.column.set("Magnification", mag)
    return tw


def test_every_product_is_a_preset():
    for number in PRODUCTS:
        name = preset_name(number)
        assert name in CALIBRATION_STANDARDS
        assert from_name(name, seed=1).options["standard"] == number
        build_structure(number)


def _period_nm(img, pixel_nm, near_nm):
    """The strongest spatial period of *img* within 30% of *near_nm* (the fundamental, not a
    harmonic or the crumple), refined by a zoomed DFT around the FFT peak."""
    n = img.shape[0]
    w = np.hanning(n)
    f = (img - img.mean()) * np.outer(w, w)
    F = np.abs(np.fft.rfft2(f))
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.rfftfreq(n)[None, :]
    k = np.hypot(kx, ky) * near_nm / pixel_nm
    F[(k < 0.7) | (k > 1.3)] = 0.0
    iy, ix = np.unravel_index(np.argmax(F), F.shape)
    k0x, k0y = kx[0, ix], ky[iy, 0]
    d = np.linspace(-1.0, 1.0, 41) / n
    x = np.arange(n)
    Ex = np.exp(-2j * np.pi * np.outer(k0x + d, x))  # (41, n)
    Ey = np.exp(-2j * np.pi * np.outer(k0y + d, x))
    Z = np.abs(Ey @ f @ Ex.T)  # (ky, kx)
    jy, jx = np.unravel_index(np.argmax(Z), Z.shape)
    return pixel_nm / math.hypot(k0x + d[jx], k0y + d[jy])


@pytest.mark.parametrize("number", ["607", "607-A", "606", "677"])
def test_the_grating_period_is_the_published_one(number):
    tw = _twin(number, 2000)
    tw.column.set("Intensity", 0.95)  # a beam spread past the field: no disc edge
    tw.column.set_defocus_um(-3.0)
    req = tw.request()
    img = tw.flux(req).astype(float)
    published = PRODUCTS[number].numbers["period_nm"]
    assert _period_nm(img, tw.optics(req).specimen_pixel_nm, published) == pytest.approx(published, rel=0.005)


def test_latex_spheres_have_the_published_diameter():
    tw = _twin("610-260", 4000)
    gt = tw.ground_truth(as_arrays=True)
    chord = gt["thickness_nm"].max() - 15.0  # the carbon support under the sphere
    assert chord == pytest.approx(260.0, rel=0.04)
    assert (gt["material_id"] == int(MaterialId.PROTEIN)).mean() > 0.02


def test_the_island_film_coverage_follows_the_deposit():
    g = np.mgrid[0:300, 0:300] * 0.0008
    prev = 0.0
    for c in (0.15, 0.3, 0.5, 0.65, 0.8):
        rel = (c / 0.6) ** 1.25
        inside, t, grain = island_film(5, g[1], g[0], np.full(g[0].shape, rel), 1.0, 8.0, 0.6,
                                       int(MaterialId.GOLD))
        assert inside.mean() == pytest.approx(c, abs=0.06)
        assert inside.mean() > prev
        prev = inside.mean()
        # the metal is conserved: islands carry the deposit of their surroundings
        assert t[inside].mean() * inside.mean() == pytest.approx(rel, rel=0.1)
        assert grain.size == inside.sum()


def test_shadowing_coats_the_facing_side_and_leaves_a_shadow_behind_a_ridge():
    """Source along +x at 25 deg: behind a 20 nm ridge (x just below it) lies a shadow ~43 nm
    long; the flank that faces the source catches more metal than the flat."""
    g = WaffleGrating(1.0, depth_nm=20.0, line_fraction=0.2, ramp_nm=10.0, edge_nm=0.0, wavy_nm=0.0,
                      both=False)
    s = ShadowedReplicaStructure(g, rough_nm=0.0, crumple_nm=0.0, azimuth_deg=0.0, elevation_deg=25.0)
    x = np.linspace(0.0, 1.0, 2001)
    y = np.zeros_like(x)
    h = s._surface(0, x, y, False)

    class _B:
        X, Y = x, y

    occl = s._shadow(0, x, y, _B, h, None, 0.0005, math.tan(s.elev))
    # the ridge spans [0.4, 0.6] um; its shadow falls on x < 0.4 (the source is at +x)
    assert np.all(occl[(x > 0.375) & (x < 0.395)] > 0.9)
    assert np.all(occl[(x > 0.1) & (x < 0.3)] == 0.0)
    assert np.all(occl[(x > 0.45) & (x < 0.55)] == 0.0), "the ridge top is lit"


def test_the_shadowed_replica_has_gold_islands_at_high_magnification():
    tw = _twin("607", 16000)
    gt = tw.ground_truth(as_arrays=True)
    au = gt["material_id"] == int(MaterialId.GOLD)
    assert 0.2 < au.mean() < 0.8
    assert np.unique(gt["grain_id"][au]).size > 200, "nanocrystalline islands"
