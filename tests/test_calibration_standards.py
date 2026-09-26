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
from de_twin.specimen.fieldmap import ViewWindow
from de_twin.twin import DigitalTwin


def _centre_on(tw, kind, pixel_um, n=512):
    """Put the densest spot of a feature (``latex``, ``graphite`` edge-on shells, ``moo3``,
    ``film``: no hole) on axis: rasterize coarse windows around home, then move the stage."""
    from scipy import ndimage

    from de_twin.optics.derive import stage_for_view_center

    home = tw.specimen.home_um()
    for k in range(25):
        r = 0.0 if k == 0 else n * pixel_um * (0.5 + 0.5 * k ** 0.5)
        c = (home[0] + r * math.cos(2.4 * k), home[1] + r * math.sin(2.4 * k))
        fm = tw.specimen.rasterize(ViewWindow(center_um=c, pixel_um=pixel_um, shape=(n, n)))
        m, g = fm.material_id, fm.grain_id
        hit = {"latex": m == int(MaterialId.PROTEIN), "moo3": m == int(MaterialId.MOLYBDENUM_TRIOXIDE),
               "graphite": (m == int(MaterialId.GRAPHITE)) & (g >= 0),
               "film": ndimage.binary_erosion(m != 0, iterations=n // 4)}[kind]
        if hit.any():
            i, j = np.unravel_index(np.argmax(ndimage.uniform_filter(hit.astype(float), 9) * hit), hit.shape)
            x, y = stage_for_view_center(fm.view.pixel_to_world(i, j), tw.column.state(), tw.optics_config)
            tw.column.set_stage(x=x, y=y)
            tw.clock.advance(30.0)  # let the stage get there
            return fm
    raise AssertionError(f"no {kind} found")


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


@pytest.mark.parametrize("number", ["610-17", "610-61"])
def test_latex_spheres_have_the_published_diameter(number):
    """Nominal 0.26 um (<= 3% uniformity) and certified 200 +/- 6 nm (sd 3.4 nm): the spheres'
    chords peak at the published size, on a 15 nm carbon film (the layer under them)."""
    d = PRODUCTS[number].numbers["sphere_nm"]
    tw = _twin(number, 4000)
    _centre_on(tw, "latex", 0.004)
    gt = tw.ground_truth(as_arrays=True)
    latex = gt["material_id"] == int(MaterialId.PROTEIN)
    assert latex.mean() > 0.01
    chord = gt["thickness_nm"][latex]  # the sphere alone: its support film is the layer under it
    assert 0.94 * d < chord.max() < 1.12 * d
    assert np.allclose(gt["under_thickness_nm"][latex], 15.0) and (gt["under_material"][latex] == 1).all()


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
    long, with a soft edge; the ridge top and the open ground are lit."""
    g = WaffleGrating(1.0, depth_nm=20.0, line_fraction=0.2, ramp_nm=10.0, edge_nm=0.0, wavy_nm=0.0,
                      both=False)
    s = ShadowedReplicaStructure(g, rough_nm=0.0, crumple_nm=0.0, azimuth_deg=0.0, elevation_deg=25.0)
    x = np.linspace(0.0, 1.0, 2001)
    h, gx, gy, lit = s.tile().sample(x, np.full_like(x, 0.3))
    # the ridge spans [0.4, 0.6] um; its shadow falls on x < 0.4 (the source is at +x)
    assert np.all(lit[(x > 0.372) & (x < 0.393)] < 0.1)
    assert np.all(lit[(x > 0.1) & (x < 0.3)] > 0.99)
    assert np.all(lit[(x > 0.45) & (x < 0.55)] > 0.99), "the ridge top is lit"
    assert gx[(x > 0.39) & (x < 0.41)].max() > 1.0 and gx[(x > 0.59) & (x < 0.61)].min() < -1.0


def test_the_shadowed_replica_has_gold_islands_at_high_magnification():
    tw = _twin("607", 16000)
    gt = tw.ground_truth(as_arrays=True)
    au = gt["material_id"] == int(MaterialId.GOLD)
    assert 0.2 < au.mean() < 0.8
    assert np.unique(gt["grain_id"][au]).size > 200, "nanocrystalline islands"


def _spectrum_peak_nm(img, pixel_nm, lo_nm, hi_nm):
    """The d-spacing (nm) of the strongest power-spectrum peak between d = lo and hi."""
    n = img.shape[0]
    w = np.hanning(n)
    F = np.abs(np.fft.fftshift(np.fft.fft2((img - img.mean()) * np.outer(w, w))))
    k = np.hypot(*np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2))
    d = n * pixel_nm / np.maximum(k, 1e-9)
    F[(k == 0) | (d < lo_nm) | (d > hi_nm)] = 0.0
    iy, ix = np.unravel_index(np.argmax(F), F.shape)
    return float(d[iy, ix])


def _hrtem(number, mag, kind, pixel_um):
    tw = _twin(number, mag)
    tw.column.set("Intensity", 0.95)  # a spread, near-parallel beam
    tw.column.set_defocus_um(-0.066)  # ~Scherzer at 200 kV, Cs 1.2 mm
    _centre_on(tw, kind, pixel_um)
    req = tw.request()
    return tw.flux(req).astype(float), tw.optics(req).specimen_pixel_nm


def test_oriented_gold_foil_shows_the_published_lattice_spacings():
    img, p = _hrtem("646", 800000, "film", 0.0005)
    assert _spectrum_peak_nm(img, p, 0.18, 0.25) == pytest.approx(0.204, rel=0.02)
    assert _spectrum_peak_nm(img, p, 0.13, 0.16) == pytest.approx(0.144, rel=0.02)


def test_graphitized_carbon_black_shows_0_34_nm_fringes():
    img, p = _hrtem("645", 600000, "graphite", 0.001)
    # published "0.34 nm": graphite (002), 0.3355 nm, to two figures
    assert _spectrum_peak_nm(img, p, 0.25, 0.5) == pytest.approx(0.34, abs=0.0075)


def test_catalase_shows_its_two_lattice_spacings():
    from de_twin.optics.derive import stage_for_view_center

    tw = _twin("612", 40000)
    tw.column.set("Intensity", 0.95)
    tw.column.set_defocus_um(-0.3)
    home = tw.specimen.home_um()
    for k in range(60):  # a field inside a crystal: the stained lattice everywhere
        c = (home[0] + 0.5 * k ** 0.5 * math.cos(2.4 * k), home[1] + 0.5 * k ** 0.5 * math.sin(2.4 * k))
        fm = tw.specimen.rasterize(ViewWindow(center_um=c, pixel_um=0.00025, shape=(1024, 1024)))
        t = fm.thickness_nm.astype(float)
        if (t > np.percentile(t, 1) + 3.0).mean() > 0.3 and np.ptp(t) > 5.0:
            break
    x, y = stage_for_view_center(c, tw.column.state(), tw.optics_config)
    tw.column.set_stage(x=x, y=y)
    tw.clock.advance(30.0)
    req = tw.request()
    img, p = tw.flux(req).astype(float), tw.optics(req).specimen_pixel_nm
    assert _spectrum_peak_nm(img, p, 7.8, 10.0) == pytest.approx(8.75, rel=0.03)
    assert _spectrum_peak_nm(img, p, 6.0, 7.6) == pytest.approx(6.85, rel=0.03)


def test_a_moo3_lath_long_edge_is_its_crystal_long_axis():
    """The image / diffraction rotation standard: each lath's crystal [010] (Pnma; [001] Pbnm)
    lies along its long edge, so the edge in the image and the spots of its pattern are tied."""
    from scipy import ndimage

    from de_twin.specimen.fieldmap import _direct_basis

    tw = _twin("625", 2500)
    fm = _centre_on(tw, "moo3", 0.02)
    mid = int(MaterialId.MOLYBDENUM_TRIOXIDE)
    lab, n = ndimage.label(fm.material_id == mid)
    checked = 0
    for k in range(1, n + 1):
        one = lab == k
        g = fm.grain_id[one]
        if one.sum() < 300 or (g == g[0]).mean() < 0.99:
            continue  # small, or laths overlapping
        ring = ndimage.binary_dilation(one, iterations=3) & ~one
        if ((fm.material_id[ring] == mid) & (fm.grain_id[ring] != g[0])).any() or (fm.grain_id == g[0]).sum() > one.sum():
            continue  # another lath touches it (and may hide part of it)
        rows, cols = np.nonzero(one)
        wx, wy = fm.view.pixel_to_world(rows, cols)  # world coordinates: no raster flips
        evals, evecs = np.linalg.eigh(np.cov(np.vstack([wx, wy])))
        if evals[1] < 6.0 * evals[0]:
            continue  # cut by the window's edge
        box = 12.0 * math.sqrt(evals[0] * evals[1])  # the area of a rectangle with these moments
        if one.sum() * fm.view.pixel_um ** 2 < 0.85 * box:
            continue  # not one clean lath (overlapping laths)
        b = tw.specimen.grains.matrices[int(g[0])] @ _direct_basis(mid) @ np.array([0.0, 1.0, 0.0])
        b = b[:2] / np.linalg.norm(b[:2])
        assert abs(float(np.dot(evecs[:, 1], b))) > math.cos(math.radians(2.0))
        checked += 1
    assert checked >= 1


def test_magical_marker_sets_are_at_the_calibrated_depths():
    from de_twin.specimen.standards.magical import SET_SPACINGS_UM, SET_SPANS_NM, layer_bands

    sets = np.array(layer_bands()).reshape(4, 5, 2)
    centres = 0.5 * (sets[:, 0, 0] + sets[:, -1, 1])
    assert np.allclose(np.diff(np.concatenate([[0.0], centres])), SET_SPACINGS_UM)
    assert np.allclose((sets[:, -1, 1] - sets[:, 0, 0]) * 1000.0, SET_SPANS_NM)
    tw = _twin("675", 2000)
    fm = tw.specimen.rasterize(ViewWindow(center_um=tw.specimen.home_um(), pixel_um=0.01, shape=(1024, 1024)))
    assert (fm.material_id == int(MaterialId.SILICON_GERMANIUM)).any()


def test_lattice_fringes_are_as_strong_as_on_a_real_microscope():
    """Kinematic fringe amplitudes (times the calibrated efficiency): Si <011> in MAG*I*CAL
    shows {111} fringes of ~34 % contrast, as in Ted Pella's lattice image (not the few
    percent a fixed phase amplitude gave)."""
    from de_twin.optics import OpticsConfig

    tw = DigitalTwin(preset_name("675"), camera="DESim", clock=ManualClock(), seed=1,
                     optics_config=OpticsConfig(objective_aperture_mrad=25.0))
    tw.column.set("Magnification", 800000)
    tw.column.set("Intensity", 0.95)
    tw.column.set_defocus_um(-0.066)
    home = tw.specimen.home_um()
    from de_twin.optics.derive import stage_for_view_center

    # in the silicon, 0.6 um below the surface: above the first marker set
    x, y = stage_for_view_center((home[0], home[1] + 0.6), tw.column.state(), tw.optics_config)
    tw.column.set_stage(x=x, y=y)
    tw.clock.advance(30.0)
    gt = tw.ground_truth(as_arrays=True)
    assert (gt["material_id"] == int(MaterialId.SILICON)).mean() > 0.99
    img = tw.flux(tw.request()).astype(float)
    c = img[256:768, 256:768]
    assert 0.2 < c.std() / c.mean() < 0.6
    assert _spectrum_peak_nm(img, tw.optics(tw.request()).specimen_pixel_nm, 0.25, 0.4) == pytest.approx(0.3135, rel=0.03)
