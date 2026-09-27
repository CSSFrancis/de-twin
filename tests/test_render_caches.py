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


@pytest.mark.parametrize("name", ["Dense Au on holey C", "Ted Pella 607 - 2160 l/mm grating replica (waffle)"])
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
    again = [tem.bragg_contrast(fm, o, r.grains, r.crystallinity, r.config) for o, fm in reversed(views)][::-1]
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
