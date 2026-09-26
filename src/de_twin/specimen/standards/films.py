"""Film standards: holey carbon (609), evaporated metal islands on holey carbon (611 Pt/Ir,
613 Au, the gold of 638), evaporated polycrystalline aluminium (619) and the oriented
single-crystal gold foil (646)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ...hashing import SeedKind, hash_seed
from ..fieldmap import grain_id_from_hash
from ..geometry import NM_PER_UM
from ..materials import MaterialId
from ..noise import cells, fbm
from ..structures import _claim, _owner_pixels, _resolved
from .common import HoleyFilm, StandardStructure
from .replica import carbon_equivalent, island_film


def _seed(owner) -> int:
    return int(hash_seed(owner.seed, SeedKind.STRUCTURE, 0)) & 0xFFFFFFFF


class HoleyCarbonStructure(StandardStructure):
    """A holey carbon film, optionally with an evaporated metal island film on it (the metal
    lands on the carbon only) and particles on it (`extra`, e.g. graphitized carbon black).

    The holes' Fresnel fringes are the classic astigmatism check (609); the gaps between
    metal islands the resolution check (613, 611)."""

    def __init__(self, film: Optional[HoleyFilm] = None, metal: Optional[int] = None, deposit_nm: float = 0.0,
                 island_nm: float = 10.0, coverage: float = 0.6, grain_nm: float = 10.0, extra=None):
        self.film = film or HoleyFilm()
        self.metal = None if metal is None else int(metal)
        self.deposit, self.island_nm, self.coverage, self.grain_nm = (float(deposit_nm), float(island_nm),
                                                                      float(coverage), float(grain_nm))
        self.extra = extra

    def base_nm(self) -> float:
        return self.film.t

    def mean_nm(self) -> float:
        t = self.film.t
        if self.metal is not None:
            t += self.deposit * carbon_equivalent(self.metal)
        t *= 1.0 - self.film.hole_fraction()
        if self.extra is not None:
            t += self.extra.mean_nm()
        return t

    def grain_textures(self):
        return self.extra.grain_textures() if self.extra is not None else ((), ())

    def fill(self, ctx, owner):
        if not _resolved(self.film.pop[1][1] * 2.0, ctx):
            return  # the owner carries the mean
        seed = _seed(owner)
        base = self.film.t
        islands = self.metal is not None and self.deposit > 0 and _resolved(self.island_nm / NM_PER_UM, ctx)
        eq = carbon_equivalent(self.metal) if self.metal is not None else 0.0
        for B in _owner_pixels(ctx, owner):
            # the owner is `base` of carbon everywhere: holes remove it, a membrane thins it
            t = self.film.thickness(seed, B.lx, B.ly)
            hole = t <= 0.0
            film = ~hole
            if self.metal is not None and self.deposit > 0 and not islands:
                # unresolved islands: their deposit as the carbon that attenuates like it
                t = np.where(film, t + self.deposit * eq, 0.0)
            ctx.add_thickness(B.flat, t - base)
            if hole.any():
                ctx.material[B.sub(hole)] = MaterialId.VACUUM
            if islands:
                dep = np.where(film, self.deposit, 0.0)
                inside, t, grain = island_film(seed, B.lx, B.ly, dep, self.deposit, self.island_nm, self.coverage,
                                               self.metal, self.grain_nm)
                if inside.any():
                    f = B.sub(inside)
                    below = ctx.thick[f].astype(np.float64)
                    ctx.add_thickness(f, t[inside] + below / eq - below)  # the carbon under it as metal
                    _claim(ctx, f, grain, self.metal)
            if self.extra is not None:
                self.extra.draw(ctx, B, seed, owner)


class PolycrystalFilmStructure(StandardStructure):
    """An evaporated polycrystalline film (619: aluminium, ~31 nm): equiaxed grains `grain_nm`
    across, randomly oriented (ring patterns), shallow grooves at the grain boundaries."""

    def __init__(self, material: int = MaterialId.ALUMINUM, thickness_nm: float = 31.0, grain_nm: float = 15.0,
                 groove_nm: float = 3.0, groove_width_nm: float = 1.0):
        self.base_material = int(material)
        self.t, self.grain_nm = float(thickness_nm), float(grain_nm)
        self.groove, self.groove_w = float(groove_nm), float(groove_width_nm)

    def base_nm(self) -> float:
        return self.t

    def mean_nm(self) -> float:
        frac = min(1.0, 3.4 * self.groove_w / self.grain_nm)  # boundary area fraction (~3.4 w / d)
        return (self.t - 0.5 * self.groove * frac) * carbon_equivalent(self.base_material)

    def fill(self, ctx, owner):
        if not _resolved(self.grain_nm / NM_PER_UM, ctx):
            return
        seed = _seed(owner)
        L = self.grain_nm / NM_PER_UM
        for B in _owner_pixels(ctx, owner):
            wx = B.lx + 0.2 * L * fbm(seed ^ 0xC1, B.lx, B.ly, L, 1)
            wy = B.ly + 0.2 * L * fbm(seed ^ 0xC2, B.lx, B.ly, L, 1)
            _, gap, h = cells(seed, wx / L, wy / L, 0.0)
            groove = self.groove * np.exp(-0.5 * (gap * self.grain_nm / self.groove_w) ** 2)
            ctx.add_thickness(B.flat, -groove)
            _claim(ctx, B.flat, grain_id_from_hash(self.base_material, h), self.base_material)


class OrientedFoilStructure(StandardStructure):
    """An epitaxial single-crystal foil (646: gold, [001]): one orientation throughout (a slight
    mosaic between `domain_nm` domains gives bend-contour-like contrast), `thickness_nm` thick,
    perforated by the channels and holes of a film grown near percolation."""

    def __init__(self, material: int = MaterialId.GOLD, thickness_nm: float = 15.0, zone=(0, 0, 1),
                 mosaic_deg: float = 0.4, domain_nm: float = 250.0, hole_fraction: float = 0.12,
                 hole_nm: float = 60.0):
        self.base_material = int(material)
        self.t, self.zone, self.mosaic = float(thickness_nm), tuple(zone), float(mosaic_deg)
        self.domain_nm, self.hole_fraction, self.hole_nm = float(domain_nm), float(hole_fraction), float(hole_nm)

    def base_nm(self) -> float:
        return self.t

    def mean_nm(self) -> float:
        return self.t * (1.0 - self.hole_fraction) * carbon_equivalent(self.base_material)

    def grain_textures(self):
        from ...crystal.orientation import Texture

        return ((self.base_material, Texture("single", self.zone, self.mosaic)),), ()

    def fill(self, ctx, owner):
        if not _resolved(self.hole_nm / NM_PER_UM, ctx):
            return
        seed = _seed(owner)
        L = self.hole_nm / NM_PER_UM
        D = self.domain_nm / NM_PER_UM
        for B in _owner_pixels(ctx, owner):
            # holes: the low tail of a smooth field (rounded, elongated channels)
            n = fbm(seed ^ 0xD1, B.lx, B.ly, 2.5 * L, 3, 0.45)
            hole = n < _HOLE_THRESHOLD(self.hole_fraction)
            _, _, h = cells(seed ^ 0xD2, B.lx / D, B.ly / D, 0.0)
            ctx.add_thickness(B.flat, np.where(hole, -self.t, 0.0) + 0.6 * fbm(seed ^ 0xD3, B.lx, B.ly, 0.02, 2))
            _claim(ctx, B.flat, grain_id_from_hash(self.base_material, h), self.base_material)
            if hole.any():
                f = B.sub(hole)
                ctx.material[f] = MaterialId.VACUUM
                ctx.grain[f] = -1


def _HOLE_THRESHOLD(fraction: float) -> float:
    """The fbm (3 octaves, gain 0.45) value below which `fraction` of the plane lies."""
    table = _THRESHOLDS.get(fraction)
    if table is None:
        rng = np.random.default_rng(7)
        x, y = rng.uniform(0, 500, 30000), rng.uniform(0, 500, 30000)
        table = _THRESHOLDS[fraction] = float(np.quantile(fbm(11, x, y, 1.0, 3, 0.45), fraction))
    return table


_THRESHOLDS: dict = {}
