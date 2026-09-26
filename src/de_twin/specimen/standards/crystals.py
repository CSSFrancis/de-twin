"""Crystal standards: molybdenum trioxide laths (625, image / diffraction rotation) and
graphitized carbon black (645, and the particles of 638: 0.34 nm lattice fringes)."""

from __future__ import annotations

import math

import numpy as np

from ...hashing import SeedKind, hash_seed
from ..geometry import NM_PER_UM
from ..materials import MaterialId
from ..structures import _claim, _owner_pixels, _resolved
from .common import StandardStructure, angle_grain, angle_overrides, cell_rng, cells_near
from .replica import carbon_equivalent
from ..noise import fbm

try:
    import numba as _nb

    def _njit(**kw):
        return _nb.njit(cache=True, fastmath=True, nogil=True, **kw)

    _prange = _nb.prange
except Exception:  # noqa: BLE001
    def _njit(**kw):
        return lambda f: f

    _prange = range


def _seed(owner) -> int:
    return int(hash_seed(owner.seed, SeedKind.STRUCTURE, 0)) & 0xFFFFFFFF


class MoO3LathsStructure(StandardStructure):
    """MoO3 crystals on a carbon film: thin laths lying on their (010) face (Pbnm; (100) in the
    Pnma setting used here), elongated along [001] (Pbnm c = 0.3697 nm; Pnma b). The long
    edge of a lath is a known crystal direction, so comparing it in the image with the
    spots of its diffraction pattern measures the image / diffraction rotation."""

    def __init__(self, film_nm: float = 15.0, cell_um: float = 4.0, per_cell: float = 1.3,
                 length_um: float = 2.5, length_sigma: float = 0.45, aspect=(3.0, 7.0),
                 thickness_nm: float = 60.0, thickness_sigma: float = 0.35):
        self.film, self.cell, self.per_cell = float(film_nm), float(cell_um), float(per_cell)
        self.length, self.length_sigma, self.aspect = float(length_um), float(length_sigma), tuple(aspect)
        self.thick, self.thick_sigma = float(thickness_nm), float(thickness_sigma)
        self.max_half = 4.0  # um

    def base_nm(self) -> float:
        return self.film

    def mean_nm(self) -> float:
        L = self.length * math.exp(0.5 * self.length_sigma ** 2)
        w = L / (0.5 * sum(self.aspect))
        t = self.thick * math.exp(0.5 * self.thick_sigma ** 2)
        eq = carbon_equivalent(MaterialId.MOLYBDENUM_TRIOXIDE)
        return self.film + self.per_cell / self.cell ** 2 * L * w * t * eq

    def grain_textures(self):
        # zone [100] (Pnma) along the beam, the long axis [010] (Pnma) at the grain's angle
        return (), angle_overrides(MaterialId.MOLYBDENUM_TRIOXIDE, (1, 0, 0), (0, 1, 0), 0.3)

    def laths(self, seed, i, j):
        r = cell_rng(seed, 0x625, i, j)
        for _ in range(r.poisson(self.per_cell)):
            cx, cy = (i + r.uniform()) * self.cell, (j + r.uniform()) * self.cell
            ang = r.uniform(0.0, math.pi)
            L = min(r.lognormal(self.length, self.length_sigma), 2.0 * self.max_half)
            W = max(L / r.uniform(*self.aspect), 0.06)
            t = min(max(r.lognormal(self.thick, self.thick_sigma), 12.0), 150.0)
            steps = (r.uniform(-0.3, 0.3), r.uniform(0.0, 1.0), r.uniform(0.1, 0.25))  # a terrace
            # the ends are cut by crystal faces at an angle (a facetted, sword-like tip)
            yield cx, cy, ang, L, W, t, steps, (r.uniform(0.3, 1.2), r.uniform(-0.6, 0.6), r.uniform(0.3, 1.2), r.uniform(-0.6, 0.6))

    def fill(self, ctx, owner):
        if not _resolved(0.05, ctx):
            return
        seed = _seed(owner)
        mean = self.mean_nm()
        eq = carbon_equivalent(MaterialId.MOLYBDENUM_TRIOXIDE)
        for B in _owner_pixels(ctx, owner):
            ctx.add_thickness(B.flat, self.film - mean)
            lx, ly = B.lx, B.ly
            for i, j in cells_near(lx, ly, self.cell, self.max_half):
                for cx, cy, ang, L, W, t, (sx, sp, sh), (k1, o1, k2, o2) in self.laths(seed, i, j):
                    c, s = math.cos(ang), math.sin(ang)
                    u = (lx - cx) * c + (ly - cy) * s
                    v = -(lx - cx) * s + (ly - cy) * c
                    # straight sides; each end cut by two faces: u < L/2 - k |v - o W/2|
                    vv = v / (0.5 * W)
                    inside = (np.abs(v) < 0.5 * W)
                    inside &= u < 0.5 * L - k1 * 0.5 * W * np.abs(vv - o1)
                    inside &= -u < 0.5 * L - k2 * 0.5 * W * np.abs(vv - o2)
                    if not inside.any():
                        continue
                    # one terrace: a step across the lath at `sx` of its length
                    tt = t * (1.0 + np.where(u[inside] / L > sx, sh if sp > 0.5 else -sh, 0.0))
                    f = B.sub(inside)
                    ctx.add_thickness(f, tt - self.film + self.film / eq)
                    g = angle_grain(MaterialId.MOLYBDENUM_TRIOXIDE, ang + owner.rot)
                    _claim(ctx, f, np.full(f.shape, g, np.int32), MaterialId.MOLYBDENUM_TRIOXIDE)


@_njit(parallel=True)
def _onions_nb(x, y, cx, cy, r, ph, start, members, x0, y0, inv, nx, ny, core, t_out, q_out, a_out):
    """Per pixel, the particle it sees most of: projected shell thickness (um), its radial
    position q = d / R and the radial direction (rad). Particles are slightly irregular
    (low harmonics of the outline, phases `ph`)."""
    for k in _prange(x.size):
        i = int(math.floor((x[k] - x0) * inv))
        j = int(math.floor((y[k] - y0) * inv))
        best = 0.0
        bq = 0.0
        ba = 0.0
        tsum = 0.0
        if 0 <= i < nx and 0 <= j < ny:
            b = j * nx + i
            for m in range(start[b], start[b + 1]):
                s = members[m]
                dx = x[k] - cx[s]
                dy = y[k] - cy[s]
                d = math.sqrt(dx * dx + dy * dy)
                a = math.atan2(dy, dx)
                R = r[s] * (1.0 + 0.08 * math.cos(2.0 * a + ph[s]) + 0.05 * math.cos(3.0 * a + 2.7 * ph[s]))
                if d >= R:
                    continue
                q = d / R
                c2 = core * core
                t = 2.0 * R * (math.sqrt(max(1.0 - q * q, 0.0)) - math.sqrt(max(c2 - q * q, 0.0)))
                tsum += t
                if t > best:
                    best, bq, ba = t, q, a
        t_out[k] = tsum
        q_out[k] = bq
        a_out[k] = ba


class CarbonBlackParticles:
    """Graphitized carbon black (645, and the particles of 638): roughly round, slightly
    irregular particles 20-50 nm across with a hollow core, fused into branched aggregates.
    Their graphite (002) layers wrap the particle like an onion: seen from above they are
    edge-on near the rim, where they show 0.34 nm fringes that follow the particle's contour
    (curved, and a little wavy: turbostratic), and face-on (no fringes) towards the middle.

    Aggregates are diffusion-limited clusters (:func:`..stamps.aggregates`), turned and
    scaled at random."""

    def __init__(self, radius_nm: float = 14.0, sigma: float = 0.25, cluster_cell_um: float = 1.0,
                 cluster_fraction: float = 0.6, per_cluster: float = 40.0, core_frac: float = 0.3,
                 edge_on_frac: float = 0.55, wave_rad: float = 0.25):
        self.R, self.sigma = float(radius_nm), float(sigma)
        self.cell, self.frac, self.per = float(cluster_cell_um), float(cluster_fraction), float(per_cluster)
        self.core, self.edge, self.wave = float(core_frac), float(edge_on_frac), float(wave_rad)

    def mean_nm(self) -> float:
        R = self.R * math.exp(self.sigma ** 2)
        v = 4.0 / 3.0 * math.pi * R ** 3 * (1 - self.core ** 3)  # nm^3 per particle
        n = self.frac * self.per / (self.cell * NM_PER_UM) ** 2
        return n * v * carbon_equivalent(MaterialId.GRAPHITE)

    def grain_textures(self):
        # zone [100] along the beam, c (the layer normal) in the plane at the grain's angle
        return (), angle_overrides(MaterialId.GRAPHITE, (1, 0, 0), (0, 0, 1), 1.0)

    def particles(self, seed, i, j):
        """(x um, y um, R um, phase) of the particles of aggregate cell (i, j)."""
        from .. import stamps

        r = cell_rng(seed, 0x645, i, j)
        if r.uniform() > self.frac:
            return []
        shapes = stamps.aggregates()
        pts = shapes[int(r.uniform(0, len(shapes)))]
        n = int(min(len(pts), max(3, r.normal(self.per, 0.35 * self.per))))
        R0 = r.lognormal(self.R, 0.15) / NM_PER_UM
        a = r.uniform(0, 2 * math.pi)
        c, s_ = math.cos(a), math.sin(a)
        flip = -1.0 if r.uniform() < 0.5 else 1.0
        x0, y0 = (i + r.uniform()) * self.cell, (j + r.uniform()) * self.cell
        out = []
        for px, py in pts[:n]:
            px, py = px * flip, py
            R = min(r.lognormal(R0, self.sigma), 2.5 * R0)
            # fused: centres a little closer than touching
            out.append((x0 + 0.85 * R0 * (c * px - s_ * py), y0 + 0.85 * R0 * (s_ * px + c * py), R,
                        r.uniform(0, 2 * math.pi)))
        return out

    def draw(self, ctx, B, seed, owner):
        from ..spheres import SphereSet

        if not _resolved(0.5 * self.R / NM_PER_UM, ctx):
            return
        lx, ly = B.lx, B.ly
        span = 40.0 * self.R / NM_PER_UM  # an aggregate reaches ~30 radii from its seed
        parts = [p for i, j in cells_near(lx, ly, self.cell, span) for p in self.particles(seed, i, j)]
        if not parts:
            return
        P = np.array(parts)
        ss = SphereSet(P[:, 0], P[:, 1], P[:, 2] * 1.14)  # bins wide enough for the irregular outline
        x = np.ascontiguousarray(lx, np.float64)
        y = np.ascontiguousarray(ly, np.float64)
        t, q, ang = np.zeros(x.size), np.zeros(x.size), np.zeros(x.size)
        _onions_nb(x, y, ss.cx, ss.cy, P[:, 2].astype(np.float64), P[:, 3].astype(np.float64), ss.start,
                   ss.members, ss.x0, ss.y0, 1.0 / ss.bin, ss.nx, ss.ny, self.core, t, q, ang)
        on = t > 0
        if not on.any():
            return
        f = B.sub(on)
        tt = t[on] * NM_PER_UM
        below = ctx.thick[f].astype(np.float64)
        film = ctx.material[f] != 0
        ctx.add_thickness(f, tt - below)  # the particle on top ...
        if film.any():
            ctx.add_under(f[film], MaterialId.AMORPHOUS_CARBON, below[film])  # ... the film under it
        # the layer normal is radial; turbostratic layers wave a little about it
        wave = self.wave * fbm(seed ^ 0x646, lx[on], ly[on], 0.004, 2)
        g = np.where(q[on] > self.edge, angle_grain(MaterialId.GRAPHITE, ang[on] + wave + owner.rot), -1)
        _claim(ctx, f, g.astype(np.int32), MaterialId.GRAPHITE)


def _mask(shape, idx):
    m = np.zeros(shape, bool)
    m[idx] = True
    return m

