"""Caches of the live-view path (stage moves, magnification steps) give the image a fresh
render gives."""

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.crystal import effective_matrices, library_for
from de_twin.render import tem
from de_twin.render.diffraction import THICKNESS_BIN_NM, bragg_fraction, thickness_bin_for
from de_twin.specimen.materials import GRAINS_PER_MATERIAL
from de_twin.twin import DigitalTwin


def _dense_loss(fm, optics, grains, crystallinity, cfg):
    """The Bragg loss per pixel from a dense (every grain x every thickness bin) excitation."""
    t = fm.thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
    gid = fm.grain_id.astype(np.int64)
    gid = np.where((gid >= 0) & (gid < len(grains)) & (gid // GRAINS_PER_MATERIAL == fm.material_id), gid, -1)
    loss = np.zeros(gid.shape, np.float32)
    rows, cols = np.nonzero(gid >= 0)
    g_u, g_inv = np.unique(gid[rows, cols], return_inverse=True)
    tb_u, t_inv = np.unique(thickness_bin_for(t[rows, cols]), return_inverse=True)
    cr = np.asarray(crystallinity(g_u), float)
    lam = optics.wavelength_nm
    g_obj = optics.objective_aperture_mrad / (1000.0 * lam) if optics.objective_aperture_mrad > 0 else 0.0
    table = np.zeros((len(g_u), len(tb_u)))
    for mid in np.unique(g_u // GRAINS_PER_MATERIAL):
        sel = np.flatnonzero(g_u // GRAINS_PER_MATERIAL == mid)
        lib = library_for(int(mid), cfg.max_g_inv_nm)
        if lib is None:
            continue
        m = effective_matrices(grains.matrices[g_u[sel]], optics.alpha_rad, optics.beta_rad)
        ex = lib.excite(m, lam, np.broadcast_to(tb_u * THICKNESS_BIN_NM, (len(sel), len(tb_u))),
                        optics.ht_kv, optics.convergence_mrad)
        out = np.hypot(ex.gx, ex.gy) > g_obj
        p_out = np.stack([np.bincount(ex.owner, weights=ex.intensity[:, j] * out, minlength=len(sel))
                          for j in range(len(tb_u))], 1)
        table[sel] = cr[sel, None] * bragg_fraction(ex.total) / np.maximum(ex.total, 1e-300) * p_out
    loss[rows, cols] = cfg.diffraction_contrast_scale * table[g_inv, t_inv]
    return loss


GRATING = "Ted Pella 607 - 2160 l/mm grating replica (waffle)"


@pytest.mark.parametrize("name", ["Dense Au on holey C", GRATING])
def test_the_bragg_memo_is_the_dense_excitation_in_any_order(name):
    tw = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=1)
    req = tw.request()
    r = tw.renderer
    tem._BRAGG_MEMOS.clear()
    views = []
    for mag in (25000.0, 20000.0, 300000.0):
        tw.column.set("Magnification", mag)
        o = tw.optics(req)
        views.append((o, r.field_map(o)[0]))
    first = [tem.bragg_contrast(fm, o, r.grains, r.crystallinity, r.config) for o, fm in views]
    assert tem._BRAGG_MEMOS, "the excitations are kept"
    tem._BRAGG_MEMOS.clear()
    again = [tem.bragg_contrast(fm, o, r.grains, r.crystallinity, r.config)
             for o, fm in reversed(views)][::-1]
    for (o, fm), a, b in zip(views, first, again):
        np.testing.assert_array_equal(a.loss, b.loss)
        assert a.fringes.keys() == b.fringes.keys()
        for k in a.fringes:
            for x, y in zip(a.fringes[k], b.fringes[k]):
                np.testing.assert_array_equal(x, y)
        np.testing.assert_allclose(a.loss, _dense_loss(fm, o, r.grains, r.crystallinity, r.config),
                                   rtol=1e-6, atol=1e-9)


def test_a_stage_tilt_is_a_new_excitation():
    tw = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=1)
    req = tw.request()
    r = tw.renderer
    o = tw.optics(req)
    fm = r.field_map(o)[0]
    a = tem.bragg_contrast(fm, o, r.grains, r.crystallinity, r.config)
    tw.column.move_stage(alpha=4.0)
    o2 = tw.optics(req)
    b = tem.bragg_contrast(fm, o2, r.grains, r.crystallinity, r.config)
    assert not np.array_equal(a.loss, b.loss)
    np.testing.assert_allclose(b.loss, _dense_loss(fm, o2, r.grains, r.crystallinity, r.config),
                               rtol=1e-6, atol=1e-9)


# ------------------------------------------------------------------ raster reuse
def _twin(name="Dense Au on holey C", **cfg):
    from de_twin.render.config import RenderConfig

    return DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=3, render_config=RenderConfig(**cfg))


def _move(tw, dx_um, dy_um=0.0):
    s = tw.column.state().stage
    tw.column.move_stage(x=s.x_um + dx_um, y=s.y_um + dy_um)


@pytest.mark.parametrize("name", ["Dense Au on holey C", GRATING, "Negative stain on carbon"])
@pytest.mark.parametrize("dx,dy", [(0.3, 0.0), (0.0, -0.4), (-0.35, 0.3)])
def test_a_reused_field_map_is_the_fresh_raster(name, dx, dy):
    tw = _twin(name)
    req = tw.request()
    tw.flux(req)
    n = tw.optics(req).view.shape[0]
    px = tw.optics(req).view.pixel_um
    _move(tw, dx * n * px, dy * n * px)
    before = tw.renderer.rasters_reused
    tw.flux(req)
    assert tw.renderer.rasters_reused == before + 1, "the move reused the last field map"
    fm = next(reversed(tw.renderer._fieldmaps.values()))[0]
    _same_raster(fm, tw.specimen.rasterize(fm.view))


def _same_raster(fm, ref):
    """The same pixels, up to the rounding of world coordinates (a pixel centre a few ulp
    from an edge may land on the other side of it) and the painter's order where two
    particles overlap: the scene paints particles window by window, and which one claims
    an overlap pixel can depend on how a view's windows were grouped (below 1e-3 of the
    pixels, grain ids only, in the densest scenes)."""
    n = fm.material_id.size
    assert np.count_nonzero(fm.material_id != ref.material_id) <= 2e-5 * n
    assert np.count_nonzero(fm.grain_id != ref.grain_id) <= 1e-3 * n
    assert np.count_nonzero(~np.isclose(fm.thickness_nm, ref.thickness_nm, rtol=1e-5, atol=1e-2)) <= 1e-4 * n
    assert ((fm.under_thickness_nm is None) == (ref.under_thickness_nm is None)
            or not ref.under_thickness_nm.any())
    if ref.under_thickness_nm is not None and fm.under_thickness_nm is not None:
        assert np.count_nonzero(~np.isclose(fm.under_thickness_nm, ref.under_thickness_nm, rtol=1e-5,
                                            atol=1e-2)) <= 2e-5 * n


@pytest.mark.parametrize("rotation", [0.0, 0.5 * np.pi])
def test_a_turned_flipped_tilted_view_reuses_on_its_own_lattice(rotation):
    import dataclasses

    from de_twin.render import Renderer

    tw = _twin()
    o = tw.optics(tw.request())
    v = dataclasses.replace(o.view, rotation_rad=rotation, flip_x=True, cos_alpha=0.9)
    r = Renderer(tw.specimen, tw.renderer.config)
    a = r._padded_optics(dataclasses.replace(o, view=v))[0]
    r._panned_field_map(a, 0.0)
    step = 0.25 * v.shape[1] * v.pixel_um
    moved = dataclasses.replace(v, center_um=(v.center_um[0] + step, v.center_um[1] - 0.1))
    b = r._padded_optics(dataclasses.replace(o, view=moved))[0]
    fm, _ = r._panned_field_map(b, 0.0)
    assert r.rasters_reused == 1
    _same_raster(fm, tw.specimen.rasterize(fm.view))


def test_a_moved_view_is_the_image_without_reuse():
    a, b = _twin(), _twin(reuse_rasters=False)
    req = a.request()
    for tw in (a, b):
        tw.flux(req)
        n = tw.optics(req).view.shape[0]
        _move(tw, 0.3 * n * tw.optics(req).view.pixel_um, -0.2 * n * tw.optics(req).view.pixel_um)
    img_a, img_b = a.flux(req), b.flux(req)
    assert a.renderer.rasters_reused == 1 and b.renderer.rasters_reused == 0
    close = np.isclose(img_a, img_b, rtol=1e-3, atol=1e-3 * float(img_b.mean()))
    assert close.mean() > 0.999
    assert img_a.mean() == pytest.approx(img_b.mean(), rel=1e-4)


def test_a_view_is_the_same_image_however_it_was_reached():
    """World-locked and deterministic: a view reached by a drag (crops of lattice-centred
    rasters) is the fresh render of that view, away from the transfer's periodic edges."""
    a, b = _twin(), _twin()
    req = a.request()
    a.flux(req)
    n = a.optics(req).view.shape[0]
    px = a.optics(req).view.pixel_um
    for step in (0.07, 0.3, 0.07):
        _move(a, step * n * px, 0.5 * step * n * px)
        a.flux(req)
    _move(b, 0.44 * n * px, 0.22 * n * px)
    img_a, img_b = a.flux(req), b.flux(req)
    inner = (slice(60, -60), slice(60, -60))
    assert np.corrcoef(img_a[inner].ravel(), img_b[inner].ravel())[0, 1] > 0.995
    assert img_a[inner].mean() == pytest.approx(img_b[inner].mean(), rel=0.01)


def test_a_magnification_step_renders_the_new_view():
    tw, ref = _twin(), _twin()
    req = tw.request()
    tw.flux(req)
    built = tw.renderer.rasters_built
    for t in (tw, ref):
        t.column.set("Magnification", 25000.0)
    img = tw.flux(req)
    assert tw.renderer.rasters_built == built + 1 and tw.renderer.rasters_reused == 0
    np.testing.assert_array_equal(img, ref.flux(req))


def test_a_jump_back_is_a_crop():
    tw = _twin()
    req = tw.request()
    home = tw.flux(req)
    _move(tw, 30.0, 20.0)
    tw.flux(req)
    _move(tw, -30.0, -20.0)
    built = tw.renderer.rasters_built
    np.testing.assert_array_equal(tw.flux(req), home)
    assert tw.renderer.rasters_built == built


# ------------------------------------------------------------------ interactive
def test_a_magnification_step_previews_then_settles_to_the_full_render():
    import time

    tw = _twin(interactive=True, settle_s=0.05)
    ref = _twin()
    req = tw.request()
    tw.flux(req)
    for t in (tw, ref):
        t.column.set("Magnification", 25000.0)
    moving = tw.flux(req)
    assert tw.renderer.previews_rendered == 1
    full = ref.flux(req)
    assert moving.shape == full.shape
    assert moving.mean() == pytest.approx(full.mean(), rel=0.05)
    time.sleep(0.06)
    np.testing.assert_array_equal(tw.flux(req), full)
    assert tw.renderer.previews_rendered == 1


def test_previews_do_not_push_out_the_field_maps_a_move_reuses():
    import time

    tw = _twin(interactive=True, settle_s=0.05)
    req = tw.request()
    tw.flux(req)
    n = tw.optics(req).view.shape[0]
    px = tw.optics(req).view.pixel_um
    for _ in range(3):
        _move(tw, 0.3 * n * px)
        tw.flux(req)  # a preview
        time.sleep(0.06)
        tw.flux(req)  # the full render, from the last full field map plus a strip
    assert tw.renderer.rasters_reused >= 3


def test_serve_interactive_flag():
    import argparse

    from de_twin.cli import _build_twin

    args = argparse.Namespace(specimen="Dense Au on holey C", camera="DESim", seed=0, holder="none",
                              time_scale=1.0, mirror_temchannel=None, corrector="none", interactive=True)
    assert _build_twin(args).renderer.config.interactive
    args.interactive = False
    assert not _build_twin(args).renderer.config.interactive


@pytest.mark.parametrize("dx_px,dy_px", [(7.4, -3.3), (0.5, 0.25), (-120.7, 60.45)])
def test_a_crop_is_where_a_fresh_render_puts_the_view(dx_px, dy_px):
    """Inside the margin a move is a crop, interpolated to a fraction of a pixel: the image
    sits where a fresh render of the view puts it (not on the nearest raster pixel)."""
    from test_realism_calibrations import _shift_px

    tw, ref = _twin(), _twin()
    req = tw.request()
    tw.flux(req)
    built = tw.renderer.rasters_built
    px = tw.optics(req).view.pixel_um
    for t in (tw, ref):
        _move(t, dx_px * px, dy_px * px)
    crop = tw.flux(req)
    assert tw.renderer.rasters_built == built, "a crop"
    fresh = ref.flux(req)
    inner = (slice(64, -64), slice(64, -64))
    # (the whitened phase correlation weighs the pixel-scale texture, which a sub-pixel shift
    # of the exit wave and one of the image move a little differently)
    assert np.abs(_shift_px(fresh[inner].astype(float), crop[inner].astype(float))).max() <= 0.2
    assert np.corrcoef(crop[inner].ravel(), fresh[inner].ravel())[0, 1] > 0.999


# ------------------------------------------------------------------ strips
@pytest.mark.parametrize("name", ["Dense Au on holey C", GRATING, "Negative stain on carbon"])
def test_a_raster_in_parallel_strips_is_the_serial_raster(name):
    from de_twin.specimen.model import _numba_threadsafe

    tw = _twin(name)
    req = tw.request()
    tw.flux(req)  # runs the numba kernels (the threading layer is known after)
    v = tw.renderer._padded_optics(tw.optics(req))[0].view
    serial = tw.specimen.rasterize(v)
    strips = tw.specimen.rasterize(v, threads=4)
    if _numba_threadsafe():
        assert tw.specimen.last_stats.get("strips") == 4
    _same_raster(strips, serial)


def test_a_piece_of_a_view_is_drawn_as_the_whole_view():
    """The scene populates a placement area or draws its aggregate by how large it is in
    the view; a piece of a view (a strip, the part a stage move brings in) must decide as
    the whole view does."""
    from de_twin.render.renderer import _sub_view
    from de_twin.specimen import Specimen, ViewWindow, from_name

    s = Specimen(from_name("Dense Au on holey C"))
    v = ViewWindow(center_um=s.features()[0].center_um, pixel_um=0.1, shape=(1344, 1344))
    whole = s.rasterize(v)
    piece = s.rasterize(_sub_view(v, 544, 800, 544, 800), span_px=1344)
    alone = s.rasterize(_sub_view(v, 544, 800, 544, 800))
    np.testing.assert_array_equal(piece.material_id, whole.material_id[544:800, 544:800])
    assert not np.array_equal(alone.material_id, piece.material_id), "a small view alone decides otherwise"


def test_a_rotated_view_is_rasterised_whole():
    import dataclasses

    from de_twin.render import Renderer

    tw = _twin()
    o = tw.optics(tw.request())
    v = dataclasses.replace(o.view, rotation_rad=0.7)
    r = Renderer(tw.specimen, tw.renderer.config)
    r._panned_field_map(r._padded_optics(dataclasses.replace(o, view=v))[0], 0.0)
    step = 0.25 * v.shape[1] * v.pixel_um
    moved = dataclasses.replace(v, center_um=(v.center_um[0] + step, v.center_um[1]))
    fm, _ = r._panned_field_map(r._padded_optics(dataclasses.replace(o, view=moved))[0], 0.0)
    assert r.rasters_reused == 0 and "strips" not in tw.specimen.last_stats
    ref = tw.specimen.rasterize(fm.view)
    np.testing.assert_array_equal(fm.grain_id, ref.grain_id)
