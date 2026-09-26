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
                 crystal_nm: float = 30.0, lattice_stain_nm: float = 8.0, cell_um: float = 2.0,
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
                        prot = (0.55 * 0.5 * (1 + np.cos(2 * math.pi * un / self.a))
                                + 0.45 * 0.5 * (1 + np.cos(2 * math.pi * vn / self.b)))
                        t += self.lattice * (1.0 - prot)
                    else:
                        t += 0.5 * self.lattice
                    ctx.add_thickness(f, t)
            ctx.material[B.flat] = MaterialId.URANYL_STAIN


class FerritinStructure(StandardStructure):
    """Ferritin on a formvar/carbon film (608): 12 nm protein shells (low contrast, unstained)
    around ~7 nm iron cores. A core is four sub-units (the tetrad) with narrow gaps; seeing
    them resolved indicates an instrument resolution better than 1.25 nm. About one
    molecule in eight is apoferritin (no core)."""

    def __init__(self, film_nm: float = 10.0, cell_nm: float = 16.0, fill: float = 0.35, shell_nm: float = 12.0,
                 cavity_nm: float = 8.0, lobe_r_nm: float = 1.55, lobe_d_nm: float = 1.45, loaded: float = 0.88):
        self.film, self.cell, self.fill_frac = float(film_nm), float(cell_nm), float(fill)
        self.shell, self.cavity = float(shell_nm), float(cavity_nm)
        self.lobe_r, self.lobe_d, self.loaded = float(lobe_r_nm), float(lobe_d_nm), float(loaded)

    def base_nm(self) -> float:
        return self.film

    def fill(self, ctx, owner):
        if not _resolved(self.shell / NM_PER_UM, ctx):
            return
        seed = _seed(owner)
        lat = JitteredLattice(self.cell / NM_PER_UM, seed, 0x608, jitter_frac=0.7)
        pe = _eq(MaterialId.PROTEIN, MaterialId.AMORPHOUS_CARBON)
        fe = _eq(MaterialId.AMORPHOUS_CARBON, MaterialId.FERRIHYDRITE)
        for B in _owner_pixels(ctx, owner):
            hit = lat.nearest(B.lx, B.ly)
            h = hit.h
            # clustered: molecules gather in patches (a slow field; most of the film is bare)
            p = self.fill_frac * np.clip(2.5 * fbm(seed ^ 0xF1, hit.sx, hit.sy, 0.15, 2), 0.0, 2.5)
            present = uniform_from_hash(h ^ U64(0x61)) < p
            dx = (B.lx - hit.sx) * NM_PER_UM
            dy = (B.ly - hit.sy) * NM_PER_UM
            d2 = dx * dx + dy * dy
            R = 0.5 * self.shell
            on = present & (d2 < R * R)
            if not on.any():
                continue
            rc = 0.5 * self.cavity
            prot = 2.0 * (np.sqrt(np.maximum(R * R - d2, 0.0)) - np.sqrt(np.maximum(rc * rc - d2, 0.0)))
            ctx.add_thickness(B.sub(on), (prot * pe)[on])
            # the core: four lobes on a square, turned at random
            loaded = on & (uniform_from_hash(h ^ U64(0x62)) < self.loaded)
            if not loaded.any():
                continue
            phi = 2 * math.pi * uniform_from_hash(h ^ U64(0x63))
            core = np.zeros(d2.shape)
            for k in range(4):
                a = phi + 0.5 * math.pi * k
                ex = dx - self.lobe_d * np.cos(a)
                ey = dy - self.lobe_d * np.sin(a)
                core = np.maximum(core, 2.0 * np.sqrt(np.maximum(self.lobe_r ** 2 - ex * ex - ey * ey, 0.0)))
            iron = loaded & (core > 0)
            if iron.any():
                f = B.sub(iron)
                # the pixel becomes ferrihydrite: its carbon and protein as their equivalent
                below = ctx.thick[f].astype(np.float64)
                ctx.add_thickness(f, core[iron] + below * fe - below)
                _claim(ctx, f, np.full(f.shape, -1, np.int32), MaterialId.FERRIHYDRITE)
