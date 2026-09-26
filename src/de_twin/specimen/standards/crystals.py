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


class CarbonBlackParticles:
    """Graphitized carbon black: polyhedral particles (5-8 facets, 20-50 nm across) with a
    hollow core, their graphite (002) layers parallel to the facets, gathered in clusters and
    chains. Seen from above, the layers are edge-on near the rim, which shows the 0.34 nm
    fringes, parallel to each facet."""

    def __init__(self, radius_nm: float = 15.0, sigma: float = 0.3, cluster_cell_um: float = 1.0,
                 cluster_fraction: float = 0.6, per_cluster: float = 25.0, core_frac: float = 0.3,
                 edge_on_frac: float = 0.55):
        self.R, self.sigma = float(radius_nm), float(sigma)
        self.cell, self.frac, self.per = float(cluster_cell_um), float(cluster_fraction), float(per_cluster)
        self.core, self.edge = float(core_frac), float(edge_on_frac)

    def mean_nm(self) -> float:
        R = self.R * math.exp(self.sigma ** 2)
        v = 4.0 / 3.0 * math.pi * R ** 3 * (1 - self.core ** 3)  # nm^3 per particle
        n = self.frac * (self.per + 1) / (self.cell * NM_PER_UM) ** 2
        return n * v * carbon_equivalent(MaterialId.GRAPHITE)

    def grain_textures(self):
        # zone [100] along the beam, c (the layer normal) in the plane at the grain's angle
        return (), angle_overrides(MaterialId.GRAPHITE, (1, 0, 0), (0, 0, 1), 1.0)

    def particles(self, seed, i, j):
        r = cell_rng(seed, 0x645, i, j)
        if r.uniform() > self.frac:
            return
        # a branched aggregate: each particle fuses onto a random earlier one
        placed = []
        n = int(max(1, r.normal(self.per, 0.4 * self.per)))
        for k in range(n):
            R = min(r.lognormal(self.R, self.sigma), 3.0 * self.R) / NM_PER_UM
            if placed:
                px, py, pr = placed[int(r.uniform(0, len(placed)))]
                d = r.uniform(0.0, 2 * math.pi)
                x, y = px + 0.85 * (pr + R) * math.cos(d), py + 0.85 * (pr + R) * math.sin(d)
            else:
                x, y = (i + r.uniform()) * self.cell, (j + r.uniform()) * self.cell
            placed.append((x, y, R))
            yield x, y, R, int(r.uniform(5, 9)), r.uniform(0, 2 * math.pi)

    def draw(self, ctx, B, seed, owner):
        if not _resolved(0.5 * self.R / NM_PER_UM, ctx):
            return
        reach = self.cell + 12.0 * 3.0 * self.R / NM_PER_UM
        lx, ly = B.lx, B.ly
        for i, j in cells_near(lx, ly, self.cell, reach):
            for cx, cy, R, n, rot in self.particles(seed, i, j):
                dx, dy = lx - cx, ly - cy
                near = (np.abs(dx) < R) & (np.abs(dy) < R)
                if not near.any():
                    continue
                dx, dy = dx[near], dy[near]
                th = np.arctan2(dy, dx) - rot
                seg = 2 * math.pi / n
                k = np.floor(np.mod(th, 2 * math.pi) / seg)
                # the polygon's radius in this direction (facet k)
                rp = R * math.cos(math.pi / n) / np.cos(np.mod(th, seg) - 0.5 * seg)
                d = np.hypot(dx, dy)
                inside = d < rp
                if not inside.any():
                    continue
                q = d[inside] / rp[inside]
                # a hollow polyhedral shell ~ a sphere minus its core, scaled to the polygon
                Rp = rp[inside] * NM_PER_UM
                t = 2.0 * Rp * (np.sqrt(np.maximum(1 - q * q, 0.0))
                                - np.sqrt(np.maximum(self.core ** 2 - q * q, 0.0)))
                sel = np.flatnonzero(near)[inside]
                f = B.sub(_mask(near.shape, sel))
                ctx.add_thickness(f, t)
                facet = rot + (k[inside] + 0.5) * seg  # facet normal: the layers' c axis
                g = np.where(q > self.edge, angle_grain(MaterialId.GRAPHITE, facet + owner.rot), -1).astype(np.int32)
                _claim(ctx, f, g, MaterialId.GRAPHITE)


def _mask(shape, idx):
    m = np.zeros(shape, bool)
    m[idx] = True
    return m

