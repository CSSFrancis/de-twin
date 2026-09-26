"""Biological standards: negatively stained catalase crystals (612, lattice spacings 8.75 and
6.85 nm) and ferritin (608, the iron core's tetrad: resolution better than 1.25 nm)."""

from __future__ import annotations

import math

import numpy as np

from ...hashing import SeedKind, hash_seed, uniform_from_hash
from ..geometry import NM_PER_UM, JitteredLattice
from ..materials import MaterialId, absorption_lengths_nm
from ..noise import fbm
from ..structures import _claim, _owner_pixels, _resolved
from .common import StandardStructure, cell_rng, cells_near

U64 = np.uint64


def _seed(owner) -> int:
    return int(hash_seed(owner.seed, SeedKind.STRUCTURE, 0)) & 0xFFFFFFFF


def _eq(a: int, b: int) -> float:
    """nm of material *b* that attenuate like 1 nm of *a*."""
    lam = absorption_lengths_nm(200.0)
    return float(lam[int(b)] / lam[int(a)])


class CatalaseStructure(StandardStructure):
    """Negatively stained catalase crystals (612) on a carbon film: thin plate crystals whose
    lattice (spacings `a_nm` along the plate's length and `b_nm` across) shows as the stain
    that fills around the molecules; a thin, uneven stain layer over the film and a stain
    rim pooled along the crystal edges.

    A pixel holds one material, so the stained field is all uranyl stain, the carbon and
    protein under it at the stain thickness that attenuates like them."""

    base_material = MaterialId.URANYL_STAIN

    def __init__(self, a_nm: float = 8.75, b_nm: float = 6.85, film_nm: float = 10.0, stain_nm: float = 5.0,
                 crystal_nm: float = 30.0, lattice_stain_nm: float = 20.0, cell_um: float = 2.0,
                 per_cell: float = 1.2, length_um: float = 1.2, aspect=(1.5, 4.0)):
        self.a, self.b = float(a_nm), float(b_nm)
        self.film, self.stain, self.crystal, self.lattice = float(film_nm), float(stain_nm), float(crystal_nm), float(lattice_stain_nm)
        self.cell, self.per_cell, self.length, self.aspect = float(cell_um), float(per_cell), float(length_um), tuple(aspect)

    def base_nm(self) -> float:
        return self.stain + self.film * _eq(MaterialId.AMORPHOUS_CARBON, MaterialId.URANYL_STAIN)

    def mean_nm(self) -> float:
        return self.base_nm() * _eq(MaterialId.URANYL_STAIN, MaterialId.AMORPHOUS_CARBON)

    def crystals(self, seed, i, j):
        r = cell_rng(seed, 0x612, i, j)
        for _ in range(r.poisson(self.per_cell)):
            L = min(r.lognormal(self.length, 0.4), 4.0)
            yield ((i + r.uniform()) * self.cell, (j + r.uniform()) * self.cell, r.uniform(0, math.pi),
                   L, L / r.uniform(*self.aspect), r.uniform(0, self.a), r.uniform(0, self.b))

    def fill(self, ctx, owner):
        seed = _seed(owner)
        base = self.base_nm()
        pe = _eq(MaterialId.PROTEIN, MaterialId.URANYL_STAIN)
        ce = _eq(MaterialId.AMORPHOUS_CARBON, MaterialId.URANYL_STAIN)
        if not _resolved(0.05, ctx):
            return
        lattice = _resolved(self.b / NM_PER_UM, ctx)
        for B in _owner_pixels(ctx, owner):
            lx, ly = B.lx, B.ly
            # the stain over the film: uneven, drying into thicker patches
            ctx.add_thickness(B.flat, self.stain * 0.6 * fbm(seed ^ 0xE1, lx, ly, 0.4, 3))
            for i, j in cells_near(lx, ly, self.cell, 2.0):
                for cx, cy, ang, L, W, pu, pv in self.crystals(seed, i, j):
                    c, s = math.cos(ang), math.sin(ang)
                    u = (lx - cx) * c + (ly - cy) * s
                    v = -(lx - cx) * s + (ly - cy) * c
                    # a plate with slightly irregular straight edges
                    du = 0.5 * L - np.abs(u)
                    dv = 0.5 * W - np.abs(v)
                    edge = np.minimum(du, dv)
                    inside = edge > 0
                    rim = (edge > -0.03) & ~inside
                    if rim.any():  # stain pooled along the edge
                        ctx.add_thickness(B.sub(rim), 6.0 * np.exp(edge[rim] / 0.012))
                    if not inside.any():
                        continue
                    f = B.sub(inside)
                    t = np.full(f.shape, self.crystal * pe)
                    if lattice:
                        un = u[inside] * NM_PER_UM + pu
                        vn = v[inside] * NM_PER_UM + pv
                        # protein density: molecules on the lattice; stain fills around them
                        # both plane sets, the 8.75 nm a little stronger (Ted Pella's image:
                        # fringe amplitudes 2.8 % and 2.4 %, at right angles)
                        prot = (0.55 * 0.5 * (1 + np.cos(2 * math.pi * un / self.a))
                                + 0.45 * 0.5 * (1 + np.cos(2 * math.pi * vn / self.b)))
                        # the stain penetrates unevenly: the lattice fades in and out
                        depth = 0.55 + 0.45 * fbm(seed ^ 0xE3, lx[inside], ly[inside], 0.12, 2)
                        t += self.lattice * np.clip(depth, 0.1, 1.0) * (1.0 - prot)
                        t += 4.0 * fbm(seed ^ 0xE4, lx[inside], ly[inside], 0.25, 2)  # thicker, thinner stain
                    else:
                        t += 0.5 * self.lattice
                    ctx.add_thickness(f, t)
            ctx.material[B.flat] = MaterialId.URANYL_STAIN


class FerritinStructure(StandardStructure):
    """Ferritin on a formvar/carbon film (608): 12 nm protein shells (low contrast, unstained)
    around ~7 nm iron cores made of four sub-units (the tetrad) with ~1.25 nm gaps: resolving
    them indicates better than 1.25 nm. "Individual ferritin particles are scattered
    throughout the specimen. Aggregates of particles are also present" (Ted Pella): here
    diffusion-limited aggregates of touching molecules, and sparse singles. About one
    molecule in eight is apoferritin (no core)."""

    def __init__(self, film_nm: float = 10.0, shell_nm: float = 12.0, cavity_nm: float = 8.0,
                 lobe_r_nm: float = 1.55, lobe_d_nm: float = 1.45, loaded: float = 0.88,
                 cluster_cell_um: float = 0.35, cluster_fraction: float = 0.5, per_cluster: float = 60.0,
                 single_per_um2: float = 60.0):
        self.film, self.shell, self.cavity = float(film_nm), float(shell_nm), float(cavity_nm)
        self.lobe_r, self.lobe_d, self.loaded = float(lobe_r_nm), float(lobe_d_nm), float(loaded)
        self.cell, self.frac, self.per = float(cluster_cell_um), float(cluster_fraction), float(per_cluster)
        self.singles = float(single_per_um2)

    def base_nm(self) -> float:
        return self.film

    def molecules(self, seed, i, j):
        """(x um, y um, hash) of the molecules of cell (i, j): an aggregate and singles."""
        from .. import stamps

        r = cell_rng(seed, 0x608, i, j)
        out = []
        if r.uniform() < self.frac:
            shapes = stamps.aggregates()
            pts = shapes[int(r.uniform(0, len(shapes)))]
            n = int(min(len(pts), max(3, r.normal(self.per, 0.5 * self.per))))
            a = r.uniform(0, 2 * math.pi)
            c, s_ = math.cos(a), math.sin(a)
            x0, y0 = (i + r.uniform()) * self.cell, (j + r.uniform()) * self.cell
            R = 0.5 * self.shell / NM_PER_UM * 1.02  # touching shells
            for px, py in pts[:n]:
                out.append((x0 + R * (c * px - s_ * py), y0 + R * (s_ * px + c * py), r.hash()))
        for _ in range(r.poisson(self.singles * self.cell * self.cell)):
            out.append(((i + r.uniform()) * self.cell, (j + r.uniform()) * self.cell, r.hash()))
        return out

    def fill(self, ctx, owner):
        from ..spheres import SphereSet

        if not _resolved(self.shell / NM_PER_UM, ctx):
            return
        seed = _seed(owner)
        pe = _eq(MaterialId.PROTEIN, MaterialId.AMORPHOUS_CARBON)
        reach = 0.5 * self.cell + 16.0 * 0.5 * self.shell / NM_PER_UM
        for B in _owner_pixels(ctx, owner):
            lx, ly = B.lx, B.ly
            mol = [m for i, j in cells_near(lx, ly, self.cell, reach) for m in self.molecules(seed, i, j)]
            if not mol:
                continue
            M = np.array([(x, y) for x, y, _ in mol])
            H = np.array([h for _, _, h in mol], np.uint64)
            R, rc = 0.5 * self.shell / NM_PER_UM, 0.5 * self.cavity / NM_PER_UM
            shell = SphereSet(M[:, 0], M[:, 1], np.full(len(M), R)).top_chord(lx, ly)[1]
            cav = SphereSet(M[:, 0], M[:, 1], np.full(len(M), rc)).top_chord(lx, ly)[1]
            prot = (shell - cav) * NM_PER_UM
            on = prot > 0
            if on.any():  # protein, as carbon that attenuates like it, on the film
                ctx.add_thickness(B.sub(on), prot[on] * pe)
            # the cores: four lobes on a square, turned at random, in loaded molecules
            loaded = uniform_from_hash(H ^ U64(0x62)) < self.loaded
            phi = 2 * math.pi * uniform_from_hash(H ^ U64(0x63))
            lobes = []
            for k in range(4):
                a = phi + 0.5 * math.pi * k
                lobes.append(np.stack([M[:, 0] + self.lobe_d / NM_PER_UM * np.cos(a),
                                       M[:, 1] + self.lobe_d / NM_PER_UM * np.sin(a)], 1)[loaded])
            L = np.concatenate(lobes)
            if not len(L):
                continue
            core = SphereSet(L[:, 0], L[:, 1], np.full(len(L), self.lobe_r / NM_PER_UM)).top_chord(lx, ly)[1] * NM_PER_UM
            iron = core > 0
            if iron.any():
                f = B.sub(iron)
                below = ctx.thick[f].astype(np.float64)
                ctx.add_thickness(f, core[iron] - below)  # the core on top ...
                ctx.add_under(f, MaterialId.AMORPHOUS_CARBON, below)  # ... film and protein under it
                _claim(ctx, f, np.full(f.shape, -1, np.int32), MaterialId.FERRIHYDRITE)
