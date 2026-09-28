"""numba kernels for the specimen rasteriser (:mod:`.raster` and the structures' fills).

Each kernel evaluates, pixel by pixel, the same float64 expressions in the same order as
the NumPy code it replaces, so point-sampled geometry comes out identical (transcendental
functions may differ from NumPy's vectorised ones in the last bit). The NumPy code stays as
the reference and the fallback (``AVAILABLE`` False without numba); tests compare the two.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import numba as nb

    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False


def njit(**kw):
    if not AVAILABLE:
        return lambda f: f
    return nb.njit(cache=True, nogil=True, error_model="numpy", **kw)


prange = nb.prange if AVAILABLE else range

# Shape / Profile codes (raster.Shape, raster.Profile)
ELLIPSE, ROUGH_ELLIPSE, FACETED, RECT, RING = 0, 1, 2, 3, 4
FLAT, SPHERICAL, WEDGE, CURTAIN = 0, 1, 2, 3


@njit()
def _span(A0, Ac, Ar, B0, Bc, Br, row, rect, nx):
    """`raster.row_spans` for one row: inclusive column bounds (hi < lo: empty)."""
    a = A0 + Ar * row
    b = B0 + Br * row
    if rect:
        lo = -np.inf
        hi = np.inf
        for k in range(2):
            k0 = a if k == 0 else b
            kc = Ac if k == 0 else Bc
            if abs(kc) < 1e-300:
                if abs(k0) > 1.0:
                    lo = np.inf
            else:
                c1 = (-1.0 - k0) / kc
                c2 = (1.0 - k0) / kc
                lo = max(lo, min(c1, c2))
                hi = min(hi, max(c1, c2))
    else:
        qa = Ac * Ac + Bc * Bc
        qb = 2.0 * (a * Ac + b * Bc)
        qc = a * a + b * b - 1.0
        disc = qb * qb - 4.0 * qa * qc
        sq = math.sqrt(max(disc, 0.0))
        if disc >= 0:
            lo = (-qb - sq) / (2.0 * qa)
            hi = (-qb + sq) / (2.0 * qa)
        else:
            lo = np.inf
            hi = -np.inf
    lo = int(min(max(np.ceil(lo - 1e-9), 0.0), float(nx)))
    hi = int(min(max(np.floor(hi + 1e-9), -1.0), float(nx - 1)))
    return lo, hi


@njit(parallel=True)
def paint_large(thick, material, grain, nx, r0, r1, co, rect, ring_inner2, profile, t0, mat, gid, overwrite):
    """`raster.paint_large` for FLAT / SPHERICAL / WEDGE profiles: every row of the window's
    span, the primitive's thickness added and the material / grain claimed (`apply`)."""
    A0, Ac, Ar, B0, Bc, Br = co[0], co[1], co[2], co[3], co[4], co[5]
    for r in prange(r0, r1):
        lo, hi = _span(A0, Ac, Ar, B0, Bc, Br, float(r), rect, nx)
        for c in range(lo, hi + 1):
            lx = A0 + Ac * float(c) + Ar * float(r)
            ly = B0 + Bc * float(c) + Br * float(r)
            r2 = lx * lx + ly * ly
            if profile == FLAT:
                t = t0
            elif profile == SPHERICAL:
                t = t0 * math.sqrt(max(0.0, 1.0 - r2))
            else:
                t = t0 * min(max(0.5 * (ly + 1.0), 0.0), 1.0)
            if not t > 0.0:
                continue
            if ring_inner2 >= 0.0 and not r2 >= ring_inner2:
                continue
            f = r * nx + c
            existing = np.float64(thick[f])
            wasvac = material[f] == 0
            thick[f] = np.float32(min(max(existing + t, 0.0), 65535.0))
            if wasvac or (overwrite and t >= existing):
                material[f] = mat
            if gid >= 0:
                grain[f] = gid


# ------------------------------------------------------------------ shared scalar helpers
@njit()
def interp1(x, xp, fp):
    """``np.interp(x, xp, fp)`` for one x (NumPy's algorithm: last xp[j] <= x, exact nodes)."""
    n = xp.shape[0]
    if x < xp[0]:
        return fp[0]
    if x > xp[n - 1]:
        return fp[n - 1]
    lo = 0
    hi = n
    while lo < hi:  # last index with xp[j] <= x
        mid = lo + ((hi - lo) >> 1)
        if x >= xp[mid]:
            lo = mid + 1
        else:
            hi = mid
    j = lo - 1
    if j == n - 1:
        return fp[j]
    if xp[j] == x:
        return fp[j]
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    res = slope * (x - xp[j]) + fp[j]
    if np.isnan(res):
        res = slope * (x - xp[j + 1]) + fp[j + 1]
        if np.isnan(res) and fp[j] == fp[j + 1]:
            res = fp[j]
    return res


@njit()
def rint(x):
    """``np.rint``: to the nearest integer, halves to even."""
    r = math.floor(x + 0.5)
    if r - x == 0.5 and r % 2.0 != 0.0:
        r -= 1.0
    return r


@njit()
def _sm(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit()
def grain_from_hash(material, h, gpm):
    """`fieldmap.grain_id_from_hash` for one hash: hash_seed(h, GRAIN, 0)."""
    k = _sm(_sm(_sm(_sm(h) ^ np.uint64(6)) ^ np.uint64(0)) ^ np.uint64(0))
    return np.int32(material * gpm + np.int64(k % np.uint64(gpm)))


@njit()
def add_thickness1(thick, f, nm):
    thick[f] = np.float32(min(max(np.float64(thick[f]) + nm, 0.0), 65535.0))


@njit()
def add_under1(uthick, umat, f, material, nm):
    v = np.float32(nm)
    uthick[f] = max(uthick[f] + v, np.float32(0.0))
    if v > 0:
        umat[f] = material


@njit()
def level_offset1(c, cov1, covh, covs, lvx, levels, table, offsets):
    """`stamps._level_offset` for one coverage: (family level, threshold offset). ``covs`` is
    ``cov[1:half + 1]``, ``lvx`` ``arange(1, half + 1)``, ``cov1, covh`` = ``cov[1], cov[half]``."""
    half = lvx.shape[0]
    nlev = levels.shape[0]
    lv = interp1(min(max(c, cov1), covh), covs, lvx)
    row = int(min(max(rint((lv - 1.0) / (half - 1.0) * (nlev - 1)), 0.0), float(nlev - 1)))
    return levels[row], interp1(c, table[row], offsets)


# ------------------------------------------------------------------ polycrystalline films
@njit(parallel=True)
def polycrystal_film(thick, material, grain, nx, ra, rb, c0, c1, W, O, S1, C1, N1, A1, S2, C2, N2, A2,
                     L, amp, cseed, groove, grain_nm, groove_w, base_material, gpm):
    """`PolycrystalFilmStructure.fill`: warped cells, grooved boundaries, one grain each."""
    for r in prange(ra, rb):
        for c in range(c0, c1):
            X = W[0] + W[2] * float(c) + W[4] * float(r)
            Y = W[1] + W[3] * float(c) + W[5] * float(r)
            dx = X - O[0]
            dy = Y - O[1]
            lx = dx * O[2] - dy * O[3]
            ly = dx * O[3] + dy * O[2]
            if not (abs(lx) <= O[4] and abs(ly) <= O[5]):
                continue
            f = r * nx + c
            wx = lx + amp * fbm1(lx, ly, S1, C1, N1, A1)
            wy = ly + amp * fbm1(lx, ly, S2, C2, N2, A2)
            _, gap, h = cell1(wx / L, wy / L, cseed, 0.0, 0.8)
            g = gap * grain_nm / groove_w
            add_thickness1(thick, f, -(groove * math.exp(-0.5 * (g * g))))
            grain[f] = grain_from_hash(base_material, h, gpm)
            material[f] = base_material


# ------------------------------------------------------------------ shadowed replicas
if AVAILABLE:
    from .noise import cell1, fbm1, nearest_site_hash
    from .spheres import occlusion1, top_chord1
    from .stamps import field1, periodic1


@njit()
def _h_texture(xx, yy, FS, FC, FSN, FA, FN, crumple_on, rough_term, crumple_nm, rough_nm):
    out = 0.0
    if crumple_on:
        out = out + crumple_nm * fbm1(xx, yy, FS[4, :FN[4]], FC[4, :FN[4]], FSN[4, :FN[4]], FA[4, :FN[4]])
    if rough_term:
        out = out + rough_nm * fbm1(xx, yy, FS[5, :FN[5]], FC[5, :FN[5]], FSN[5, :FN[5]], FA[5, :FN[5]])
    return out


@njit(parallel=True)
def replica_block(thick, material, grain, uthick, umat, used, nx, ra, rb, c0, c1, W, O, FL, R, stack,
                  FS, FC, FSN, FA, FN, TX, scx, scy, sr, sstart, smem, SP, SN, D, M, I, tiles, fseed, cseed,
                  covs, lvx, levels, table, offsets):
    """`ShadowedReplicaStructure.fill` for the raster rows ra:rb, columns c0:c1 of one owner
    (the relief, the replica's texture, latex spheres, the angled metal deposit and its
    island film; see the NumPy code in `standards.replica` for the physics)."""
    has_relief, crumple_on, rough_term, has_spheres, shadowed, metal_on, islands, wavy_on, edge_on = (
        FL[0], FL[1], FL[2], FL[3], FL[4], FL[5], FL[6], FL[7], FL[8])
    wavy, edge, P, across = R[0], R[1], R[2], R[3]
    crumple_nm, rough_nm, e = TX[0], TX[1], TX[2]
    metal_nm, sin_e, cos_e, ux, uy, flat, leak, base, mean_c, ceq = (
        D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8], D[9])
    metal, sphere_mat, carbon_id, gpm = M[0], M[1], M[2], M[3]
    coverage, px_per_um, cells_per_um, sharp, G, cov1, covh = I[0], I[1], I[2], I[3], I[4], I[5], I[6]
    for r in prange(ra, rb):
        for c in range(c0, c1):
            X = W[0] + W[2] * float(c) + W[4] * float(r)
            Y = W[1] + W[3] * float(c) + W[5] * float(r)
            dx = X - O[0]
            dy = Y - O[1]
            lx = dx * O[2] - dy * O[3]
            ly = dx * O[3] + dy * O[2]
            if not (abs(lx) <= O[4] and abs(ly) <= O[5]):
                continue
            f = r * nx + c
            # the relief (looked up) and the replica's own surface (noise)
            if has_relief:
                u = lx
                v = ly
                if wavy_on:
                    u = u + wavy * fbm1(across * lx, ly, FS[0, :FN[0]], FC[0, :FN[0]], FSN[0, :FN[0]], FA[0, :FN[0]])
                    v = v + wavy * fbm1(lx, across * ly, FS[1, :FN[1]], FC[1, :FN[1]], FSN[1, :FN[1]], FA[1, :FN[1]])
                if edge_on:
                    u = u + edge * fbm1(0.3 * lx, ly, FS[2, :FN[2]], FC[2, :FN[2]], FSN[2, :FN[2]], FA[2, :FN[2]])
                    v = v + edge * fbm1(lx, 0.3 * ly, FS[3, :FN[3]], FC[3, :FN[3]], FSN[3, :FN[3]], FA[3, :FN[3]])
                fu = u / P
                fv = v / P
                h32 = periodic1(stack, 0, fu, fv)
                gx32 = periodic1(stack, 1, fu, fv)
                gy32 = periodic1(stack, 2, fu, fv)
                lit32 = periodic1(stack, 3, fu, fv)
            else:
                h32 = np.float32(0.0)
                gx32 = np.float32(0.0)
                gy32 = np.float32(0.0)
                lit32 = np.float32(1.0)
            h0 = _h_texture(lx, ly, FS, FC, FSN, FA, FN, crumple_on, rough_term, crumple_nm, rough_nm)
            hx = _h_texture(lx + e, ly, FS, FC, FSN, FA, FN, crumple_on, rough_term, crumple_nm, rough_nm)
            hy = _h_texture(lx, ly + e, FS, FC, FSN, FA, FN, crumple_on, rough_term, crumple_nm, rough_nm)
            h = np.float64(h32) + h0
            gx = np.float64(gx32) + (hx - h0) / (e * 1000.0)
            gy = np.float64(gy32) + (hy - h0) / (e * 1000.0)
            top = 0.0
            chord = 0.0
            if has_spheres:
                t_, c_ = top_chord1(lx, ly, scx, scy, sr, sstart, smem, SP[0], SP[1], SP[2], SN[0], SN[1])
                top = t_ * 1000.0
                chord = c_ * 1000.0
                if top > 0 and shadowed:
                    step, dn = SP[3], SP[4]
                    tx = top_chord1(lx + step, ly, scx, scy, sr, sstart, smem, SP[0], SP[1], SP[2], SN[0], SN[1])[0] * 1000.0
                    ty = top_chord1(lx, ly + step, scx, scy, sr, sstart, smem, SP[0], SP[1], SP[2], SN[0], SN[1])[0] * 1000.0
                    gx = min(max((tx - top) / dn, -8.0), 8.0)
                    gy = min(max((ty - top) / dn, -8.0), 8.0)
                    h = h + top
            dep = 0.0
            if metal_on:
                dep = metal_nm * max(sin_e - cos_e * (gx * ux + gy * uy), 0.0)
                dep = min(dep, 12.0 * flat)
                if has_spheres and shadowed:
                    occ = occlusion1(lx, ly, h / 1000.0, scx, scy, sr, sstart, smem, SP[0], SP[1], SP[2], SN[0],
                                     SN[1], SP[5], SP[6], SP[7], SP[8], SP[9], SP[10])
                    lit = min(np.float64(lit32), 1.0 - occ)
                    dep = dep * lit + leak * flat * (1.0 - lit)
                else:  # lit is float32 here, as in the NumPy code
                    term = np.float32(leak * flat) * (np.float32(1.0) - lit32)
                    dep = dep * np.float64(lit32) + np.float64(term)
            # layers: the film (carbon) with latex on it, the metal on top
            add_thickness1(thick, f, base - mean_c)
            latex = chord > 0
            if latex:
                add_thickness1(thick, f, chord - base)
                material[f] = sphere_mat
                add_under1(uthick, umat, f, carbon_id, base)
                used[0] = 1
            if not metal_on:
                continue
            gid = np.int32(-1)
            if islands:
                rel = dep / max(flat, 1e-9)
                cc = min(max(coverage * rel ** 0.8, 0.0), 0.9)
                lev, off = level_offset1(min(max(cc, 1e-4), 1.0 - 1e-4), cov1, covh, covs, lvx, levels, table,
                                         offsets)
                fval = field1(lx, ly, lev, tiles, fseed, px_per_um, cells_per_um, sharp) + off
                inside = fval > 0.0 and dep > 0
                t_m = dep / max(cc, 1e-3) if inside else 0.0
                if inside:
                    gid = grain_from_hash(metal, nearest_site_hash(lx / G, ly / G, cseed, 0.8), gpm)
            else:
                inside = dep > 0
                t_m = dep
            if inside:
                below = np.float64(thick[f])
                add_thickness1(thick, f, t_m - below)
                if not latex:
                    add_under1(uthick, umat, f, carbon_id, below)
                else:
                    add_under1(uthick, umat, f, sphere_mat, chord + base * ceq)
                used[0] = 1
                grain[f] = gid
                material[f] = metal
