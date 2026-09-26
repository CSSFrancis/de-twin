"""Calibration standards: Ted Pella's TEM / STEM calibration and test specimens.

Each product is a preset in :data:`CALIBRATION_STANDARDS` (``"Ted Pella 607-A - ..."``), with
its published numbers in :data:`PRODUCTS` so tests (and users) can check a calibration against
them. The physics they share lives here:

* **Shadowed replicas** (`ShadowedReplicaStructure`): a carbon replica's relief (a grating,
  latex spheres) plus a rough replica surface, shadowed with metal evaporated at an angle. The
  deposit per projected area follows the surface's slope towards the source,
  ``t = t0 (sin e - cos e (grad h . u)) / sin e`` (u: the source azimuth, e: its elevation),
  none where the relief shades the surface from the source. That is the embossed look of a
  shadowed replica: one side of every ridge dark (thick metal), the other bright.
* **Island films** (`island_film`): evaporated metal a few nm thick is not continuous but an
  island film near percolation: worm-like interconnected islands of 5-20 nm. It is a
  band-limited random field thresholded at the local coverage (which follows the local
  deposit, metal conserved), each island a crystal grain of the metal.

A pixel holds one material, so a metal island is that metal at the thickness that attenuates
like the island plus the carbon under it; where islands are smaller than two pixels the metal
is carbon-equivalent thickness, and the relief shading still shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..hashing import SeedKind, hash_seed, splitmix64, uniform_from_hash
from .fieldmap import grain_id_from_hash
from .geometry import NM_PER_UM, JitteredLattice
from .materials import MaterialId, absorption_lengths_nm
from .noise import cells, fbm
from .structures import Structure, _claim, _owner_pixels, _resolved

U64 = np.uint64


# ------------------------------------------------------------------ shared physics
def island_film(seed: int, X_um, Y_um, deposit_nm, mean_deposit_nm: float, island_nm: float,
                coverage: float, metal: int, grain_nm: float = 2.2):
    """An evaporated island film on the pixels (X, Y) (um) with local *deposit_nm*: (inside
    island, island thickness nm, grain id of each inside pixel).

    Islands grow from nucleation sites `island_nm` apart (a warped jittered grid, so their
    outlines are rounded and irregular). Where the deposit is low an island is a round
    droplet; as it rises the droplets grow until only narrow channels separate them, and
    neighbours coalesce into worm-like islands. The local coverage follows the deposit
    (``coverage`` at the mean) and the metal is conserved (island thickness = deposit /
    local coverage). Each island is nanocrystalline (grains of `grain_nm`)."""
    dep = np.asarray(deposit_nm, float)
    rel = dep / max(mean_deposit_nm, 1e-9)
    c = np.clip(coverage * rel ** 0.8, 0.0, 0.85)
    L = island_nm / NM_PER_UM
    wx = X_um + 0.28 * L * fbm(seed ^ 0xA1, X_um, Y_um, 1.1 * L, 2)
    wy = Y_um + 0.28 * L * fbm(seed ^ 0xA2, X_um, Y_um, 1.1 * L, 2)
    merge = np.clip(1.6 * (c - 0.35), 0.0, 0.7)  # coalescence past ~35% coverage
    d1, gap, _ = cells(seed, wx / L, wy / L, merge)
    # droplet radius and channel half-width (cell units) that give coverage c
    r = np.sqrt(np.minimum(c, 0.6) / math.pi) * 1.05
    half = (1.0 - c) / (3.8 * (1.0 - merge))  # the channels' area is ~3.8 x half-width per cell
    inside = (dep > 0) & (((c < 0.45) & (d1 < r)) | ((c >= 0.45) & (gap > half)))
    t = np.where(inside, dep / np.maximum(c, 1e-3), 0.0)
    grain = np.zeros(0, np.int32)
    if inside.any():
        hit = JitteredLattice(grain_nm / NM_PER_UM, seed, 0x1F).nearest(wx[inside], wy[inside])
        grain = grain_id_from_hash(metal, hit.h)
    return inside, t, grain


def carbon_equivalent(metal: int, ht_kv: float = 200.0) -> float:
    """Carbon nm that attenuates like 1 nm of *metal*."""
    lam = absorption_lengths_nm(ht_kv)
    return float(lam[int(MaterialId.AMORPHOUS_CARBON)] / lam[int(metal)])


# ------------------------------------------------------------------ reliefs
#: How much more slowly the line waviness varies across the lines than along them.
WAVY_ACROSS = 0.1


class WaffleGrating:
    """A replica cross grating: raised lines along x and y (the replica of a ruled master's
    grooves), `line_fraction` of the period wide with sloped flanks, rough edges, a long-range
    waviness that averages out (the mean period is exact) and, for the old style, defects:
    missing and doubled line segments."""

    def __init__(self, period_um: float, depth_nm: float = 30.0, line_fraction: float = 0.22,
                 ramp_nm: float = 25.0, edge_nm: float = 5.0, edge_corr_nm: float = 60.0,
                 wavy_nm: float = 15.0, wavy_um: float = 1.5, defects: float = 0.0,
                 both: bool = True, trench: bool = False):
        self.P, self.depth, self.w = float(period_um), float(depth_nm), float(line_fraction)
        self.ramp, self.edge, self.edge_corr = ramp_nm / NM_PER_UM, edge_nm / NM_PER_UM, edge_corr_nm / NM_PER_UM
        self.wavy, self.wavy_um, self.defects = wavy_nm / NM_PER_UM, float(wavy_um), float(defects)
        self.both, self.trench = bool(both), bool(trench)

    def mean_nm(self) -> float:
        f = self.w if not self.both else 1.0 - (1.0 - self.w) ** 2
        return self.depth * (1.0 - f if self.trench else f)

    def warp(self, seed, lx, ly):
        if self.wavy <= 0:
            return lx, ly
        # lines wiggle sideways: the displacement varies along a line and only slowly across
        # lines, so the pitch (and the calibration it carries) stays exact
        a = WAVY_ACROSS
        return (lx + self.wavy * fbm(seed ^ 0x51, a * lx, ly, self.wavy_um, 2),
                ly + self.wavy * fbm(seed ^ 0x52, lx, a * ly, self.wavy_um, 2))

    def height_nm(self, seed, u, v, rough: bool = True) -> np.ndarray:
        """Relief height (nm) at warped owner-local (u, v) um."""
        P, half = self.P, 0.5 * self.w * self.P
        out = None
        axes = ((u, v, 0x61), (v, u, 0x62)) if self.both else ((u, v, 0x61),)
        for a, b, salt in axes:
            line = np.floor(a / P)
            d = np.abs(a - (line + 0.5) * P)
            if rough and self.edge > 0:
                near = np.abs(d - half) < 0.5 * self.ramp + 2.0 * self.edge
                if near.any():
                    d = d.copy()
                    d[near] += self.edge * fbm(seed ^ salt, b[near], line[near] * 0.731 * P,
                                               self.edge_corr, 2)
            r = np.clip((half - d) / max(self.ramp, 1e-9) + 0.5, 0.0, 1.0)
            if rough and self.defects > 0:  # missing segments of a line (old-style gratings)
                seg = np.floor(b / (2.0 * P))
                gone = uniform_from_hash(hash_seed(seed ^ salt, SeedKind.STRUCTURE,
                                                   (line.astype(np.int64) * 100003 + seg.astype(np.int64)) & 0xFFFFFFFF,
                                                   0x33)) < self.defects
                r = np.where(gone, 0.0, r)
            out = r if out is None else np.maximum(out, r)
        h = self.depth * out
        return self.depth - h if self.trench else h


@dataclass
class LatexSpheres:
    """Polystyrene latex spheres scattered on the replica (``density`` per um^2), e.g. the
    0.261 um spheres of the 603 / 673 grating standards: polymer thickness, and relief that
    shades the metal behind them."""

    diameter_nm: float
    density_per_um2: float
    sigma_frac: float = 0.02

    def cell_um(self) -> float:
        """A lattice finer than the mean spacing, sparsely occupied, so the spheres lie at
        random rather than on a grid (but never overlap)."""
        spacing = 1.0 / math.sqrt(max(self.density_per_um2, 1e-6))
        return max(0.5 * spacing, 1.3 * self.diameter_nm / NM_PER_UM)

    def spheres(self, seed, lattice, hit):
        """(present, radius um, centre x, centre y) at the nearest lattice sites of *hit*."""
        h = hit.h
        present = (h % U64(1024)) < U64(int(min(1.0, self.density_per_um2 * self.cell_um() ** 2) * 1024))
        rj = 1.0 + self.sigma_frac * (2.0 * uniform_from_hash(splitmix64(h)) - 1.0)
        return present, 0.5 * self.diameter_nm / NM_PER_UM * rj, hit.sx, hit.sy


# ------------------------------------------------------------------ the replica structure
#: Angular size of the evaporation source seen from the specimen (radians): the penumbra.
SHADOW_PENUMBRA_RAD = math.radians(4.0)


class ShadowedReplicaStructure(Structure):
    """A shadowed carbon replica (see the module docstring).

    The replica is a conformal carbon film `base_nm` thick that follows the surface: `relief`
    (a grating, or None for a flat film), a rough "crumpled" surface (`rough_nm` over
    `rough_um`, fibrous at 10kx), a long-wavelength undulation (`crumple_nm` over
    `crumple_um`) and optional latex spheres. Projected, the film is base / cos(slope): only
    the flanks show, faintly. The contrast is the metal, evaporated at `elevation_deg` from
    `azimuth_deg`, `metal_nm` thick on a surface facing the source. It lands per projected
    area as ``metal_nm (sin e - cos e grad h . u)``, and not at all where the relief hides the
    surface: the embossed look at low magnification. At high magnification the deposit is an
    island film whose coverage follows it: dense worm-like islands where the surface faces
    the source, sparse droplets where it faces away.
    """

    required_layers = frozenset()

    def __init__(self, relief: Optional[WaffleGrating], base_nm: float = 20.0,
                 rough_nm: float = 3.0, rough_um: float = 0.06, crumple_nm: float = 12.0,
                 crumple_um: float = 0.3, metal: int = MaterialId.GOLD, metal_nm: float = 6.0,
                 elevation_deg: float = 25.0, azimuth_deg: float = 45.0, island_nm: float = 9.0,
                 coverage: float = 0.65, grain_nm: float = 2.2, shadow_leak: float = 0.12,
                 spheres: Optional[LatexSpheres] = None, sphere_material: int = MaterialId.PROTEIN):
        self.relief = relief
        self.base, self.rough_nm, self.rough_um = float(base_nm), float(rough_nm), float(rough_um)
        self.crumple_nm, self.crumple_um = float(crumple_nm), float(crumple_um)
        self.metal, self.metal_nm = int(metal), float(metal_nm)
        self.elev = math.radians(float(elevation_deg))
        az = math.radians(float(azimuth_deg))
        self.u = (math.cos(az), math.sin(az))
        self.island_nm, self.coverage, self.grain_nm = float(island_nm), float(coverage), float(grain_nm)
        self.shadow_leak = float(shadow_leak)  # metal that still reaches a shadow (source size, migration)
        self.spheres, self.sphere_material = spheres, int(sphere_material)

    def flat_deposit_nm(self) -> float:
        """Metal per projected area on a flat, level surface."""
        return self.metal_nm * math.sin(self.elev)

    def mean_carbon_nm(self) -> float:
        """Area-mean projected carbon: the film, thicker on the relief's flanks."""
        t = self.base
        if self.relief is not None:
            g = self.relief.depth / max(self.relief.ramp * NM_PER_UM, 1e-9)
            flanks = (4.0 if self.relief.both else 2.0) * self.relief.ramp / self.relief.P
            t *= 1.0 + min(flanks, 1.0) * (math.sqrt(1.0 + g * g) - 1.0)
        return t

    # -------------------------------------------------------------- the surface
    def _surface(self, seed, x, y, rough: bool, coarse: bool = False):
        """Surface height (nm) at owner-local (x, y) um, without the spheres."""
        if self.relief is not None:
            u, v = self.relief.warp(seed, x, y)
            h = self.relief.height_nm(seed, u, v, rough=not coarse)
        else:
            h = np.zeros(np.shape(x))
        if self.crumple_nm > 0:
            h = h + self.crumple_nm * fbm(seed ^ 0x93, x, y, self.crumple_um, 2)
        if rough and not coarse and self.rough_nm > 0:
            h = h + self.rough_nm * fbm(seed ^ 0x91, x, y, self.rough_um, 4, 0.6)
        return h

    def _sphere_top(self, seed, lat, X, Y):
        """(top height nm, chord nm) of the latex spheres at owner-local (X, Y) um."""
        hit = lat.nearest(X, Y)
        present, R, sx, sy = self.spheres.spheres(seed, lat, hit)
        d2 = (X - sx) ** 2 + (Y - sy) ** 2
        inside = present & (d2 < R * R)
        half = np.where(inside, np.sqrt(np.maximum(R * R - d2, 0.0)), 0.0) * NM_PER_UM
        return np.where(inside, R * NM_PER_UM + half, 0.0), 2.0 * half

    def fill(self, ctx, owner):
        seed = int(hash_seed(owner.seed, SeedKind.STRUCTURE, 0)) & 0xFFFFFFFF
        px = ctx.pixel_um
        scales = [self.crumple_um if self.crumple_nm > 0 else 1e9,
                  self.relief.P if self.relief is not None else 1e9,
                  self.spheres.diameter_nm / NM_PER_UM if self.spheres is not None else 1e9]
        if not _resolved(min(scales), ctx):
            return  # the owner carries the mean
        mean_c = self.mean_carbon_nm()
        flat = self.flat_deposit_nm()
        eq = carbon_equivalent(self.metal)
        rough = self.rough_nm > 0 and _resolved(self.rough_um / 2.0, ctx)
        islands = self.metal_nm > 0 and _resolved(self.island_nm / NM_PER_UM, ctx)
        lat = JitteredLattice(self.spheres.cell_um(), seed, 0x2F) if self.spheres is not None else None
        ux, uy = self.u
        sin_e, cos_e, tan_e = math.sin(self.elev), math.cos(self.elev), math.tan(self.elev)
        step = max(px, 1.0 / NM_PER_UM)
        dn = step * NM_PER_UM
        for B in _owner_pixels(ctx, owner):
            lx, ly = B.lx, B.ly
            h = self._surface(seed, lx, ly, rough)
            gx = (self._surface(seed, lx + step, ly, rough) - h) / dn
            gy = (self._surface(seed, lx, ly + step, rough) - h) / dn
            carbon = self.base * np.sqrt(1.0 + gx * gx + gy * gy)
            chord = np.zeros_like(h)
            if lat is not None:
                top, chord = self._sphere_top(seed, lat, lx, ly)
                on = top > 0
                if on.any():
                    tx = self._sphere_top(seed, lat, lx + step, ly)[0]
                    ty = self._sphere_top(seed, lat, lx, ly + step)[0]
                    # the sphere's own surface, its slope capped where the rim is vertical
                    gx = np.where(on, np.clip((tx - top) / dn, -8.0, 8.0), gx)
                    gy = np.where(on, np.clip((ty - top) / dn, -8.0, 8.0), gy)
                    h = np.where(on, h + top, h)
            if self.metal_nm > 0:
                dep = self.metal_nm * np.maximum(sin_e - cos_e * (gx * ux + gy * uy), 0.0)
                dep = np.minimum(dep, 2.5 * flat)
                lit = 1.0 - self._shadow(seed, lx, ly, B, h, lat, px, tan_e)
                dep = dep * lit + self.shadow_leak * flat * (1.0 - lit)
            else:
                dep = np.zeros_like(h)
            solid = carbon + chord
            latex = chord > carbon
            if not islands:
                ctx.add_thickness(B.flat, solid + dep * eq - mean_c)
                if latex.any():
                    ctx.material[B.sub(latex)] = self.sphere_material
                continue
            ctx.add_thickness(B.flat, solid - mean_c)
            if latex.any():
                ctx.material[B.sub(latex)] = self.sphere_material
            inside, t_isl, grain = island_film(seed, lx, ly, dep, flat, self.island_nm, self.coverage,
                                               self.metal, self.grain_nm)
            if inside.any():
                f = B.sub(inside)
                # the island is metal; the carbon (or latex) under it becomes metal-equivalent
                ctx.add_thickness(f, t_isl[inside] + solid[inside] / eq - solid[inside])
                _claim(ctx, f, grain, self.metal)

    def _shadow(self, seed, lx, ly, B, h0, lat, px: float, tan_e: float):
        """How much of the source the coarse relief (grating, undulation, spheres) hides from
        each pixel, 0..1: a penumbra `SHADOW_PENUMBRA_RAD` wide (the source's angular size)."""
        ux, uy = self.u
        hmax = (self.relief.depth if self.relief is not None else 0.0) + 2.0 * self.crumple_nm
        tan_lo = math.tan(max(self.elev - 0.5 * SHADOW_PENUMBRA_RAD, 0.02))
        reach = hmax / tan_lo
        dstep = max(10.0, px * NM_PER_UM)
        occl = np.zeros(h0.shape)
        if reach >= 2.0 * dstep:
            tan_hi = math.tan(self.elev + 0.5 * SHADOW_PENUMBRA_RAD)
            worst = np.full(h0.shape, -np.inf)  # steepest elevation of the horizon towards the source
            for s_nm in np.arange(dstep, reach + dstep, dstep):
                s = s_nm / NM_PER_UM
                hs = self._surface(seed, lx + ux * s, ly + uy * s, False, coarse=True)
                np.maximum(worst, (hs - h0 - 1.0) / s_nm, out=worst)
            occl = np.clip((worst - tan_lo) / (tan_hi - tan_lo), 0.0, 1.0)
        if lat is not None:
            occl = np.maximum(occl, self._sphere_shadow(seed, lat, lx, ly, h0))
        return occl

    def _sphere_shadow(self, seed, lat, lx, ly, h0):
        """Occlusion by the latex spheres, exactly: the ray from each pixel towards the source
        against the few spheres it can pass (those nearest to points along its reach)."""
        ux, uy = self.u
        ce, se = math.cos(self.elev), math.sin(self.elev)
        dx, dy, dz = ux * ce, uy * ce, se  # unit ray towards the source (um, um, um)
        Rmax = 0.5 * self.spheres.diameter_nm * (1.0 + self.spheres.sigma_frac) / NM_PER_UM
        reach = 2.0 * Rmax / max(math.tan(self.elev), 1e-3) + Rmax
        cell = lat.cell_um
        occl = np.zeros(np.shape(lx))
        seen = []
        for s in np.arange(0.0, reach + 0.5 * cell, 0.5 * cell):
            hit = lat.nearest(lx + ux * s, ly + uy * s)
            key = hit.h
            if any(np.array_equal(key, k) for k in seen):
                continue
            seen.append(key)
            present, R, sx, sy = self.spheres.spheres(seed, lat, hit)
            # the sphere rests on the film: centre height R (um); the pixel at its surface h0
            px_, py_, pz = sx - lx, sy - ly, R - h0 / NM_PER_UM
            t = px_ * dx + py_ * dy + pz * dz  # along the ray to the closest approach
            D = np.sqrt(np.maximum(px_ * px_ + py_ * py_ + pz * pz - t * t, 0.0))
            w = np.maximum(t, 1e-6) * SHADOW_PENUMBRA_RAD
            o = np.clip((R - D) / np.maximum(w, 1e-6) + 0.5, 0.0, 1.0)
            occl = np.maximum(occl, np.where(present & (t > 0.5 * R), o, 0.0))
        return occl


# ------------------------------------------------------------------ products
@dataclass(frozen=True)
class Product:
    """A Ted Pella calibration standard: what the catalogue says it is."""

    number: str
    name: str
    description: str
    use: str
    numbers: dict = field(default_factory=dict)  # published values (nm, lines/mm, ...)
    not_modelled: str = ""


PRODUCTS: dict[str, Product] = {}


def _p(number, name, description, use, numbers=None, not_modelled=""):
    PRODUCTS[number] = Product(number, name, description, use, dict(numbers or {}), not_modelled)


_p("607", "2160 l/mm diffraction grating replica (waffle)",
   "Carbon replica with Au/Pd shadowing of a crossed-line (waffle) diffraction grating, "
   "2160 lines/mm in both directions, on a G400 copper grid.",
   "Magnification calibration up to ~80-100,000x.", {"lines_per_mm": 2160.0, "period_nm": 462.9})
_p("607-A", "2160 l/mm diffraction grating replica, old style (waffle)",
   "The older-style 2160 lines/mm waffle replica with Au/Pd shadowing: more defects and a less "
   "clear pattern than the current 607, on a G400 copper grid.",
   "Aberration corrector alignment and SerialEM; magnification.", {"lines_per_mm": 2160.0, "period_nm": 462.9})
_p("606", "2160 l/mm diffraction grating replica (parallel lines)",
   "Carbon replica with Au/Pd shadowing of a parallel-line diffraction grating, 2160 lines/mm "
   "(d = 462.9 nm), on a G400 copper grid.",
   "Magnification calibration up to ~40-50,000x.", {"lines_per_mm": 2160.0, "period_nm": 462.9})
_p("603", "2160 l/mm grating replica with 0.261 um latex spheres",
   "Carbon waffle grating replica with Au/Pd shadowing and 0.261 um polystyrene latex spheres, "
   "on a G400 copper grid.",
   "A double check of magnification calibration at higher magnifications.",
   {"lines_per_mm": 2160.0, "period_nm": 462.9, "sphere_nm": 261.0})
_p("603-A", "2160 l/mm grating replica with 0.261 um latex spheres, old style",
   "The older-style 2160 lines/mm waffle replica with Au/Pd shadowing and 0.261 um latex "
   "spheres, on a G400 copper grid.",
   "Aberration corrector alignment and SerialEM; magnification double check.",
   {"lines_per_mm": 2160.0, "period_nm": 462.9, "sphere_nm": 261.0})
_p("677", "500 nm cross-line grating replica",
   "Carbon replica with Au/Pd shadowing of a cross-line grating with well-defined trench-type "
   "grooves, 500 nm pitch (2000 lines/mm) in both directions, on a G400 copper grid.",
   "Magnification calibration up to 100,000x.", {"lines_per_mm": 2000.0, "period_nm": 500.0})
_p("673", "500 nm grating replica with 261 nm latex spheres",
   "The 500 nm cross-line trench grating replica with Au/Pd shadowing and 261 nm latex spheres.",
   "Magnification double check up to 150,000x.",
   {"lines_per_mm": 2000.0, "period_nm": 500.0, "sphere_nm": 261.0})
_p("628-B", "Gold-shadowed latex",
   "0.204 um latex particles on a carbon film shadowed with a fairly heavy coating of gold.",
   "Test object for STEM (and shadow-length / resolution checks).", {"sphere_nm": 204.0})
for _d in (30, 80, 90, 170, 260, 300, 490, 1000):
    _p(f"610-{_d}", f"Polystyrene latex spheres, {_d / 1000:g} um nominal",
       f"{_d} nm polystyrene latex spheres dried on a carbon support film.",
       "Magnification test specimen.", {"sphere_nm": float(_d)})

#: Old-style gratings: rougher, wavier, with missing line segments.
_OLD = dict(edge_nm=10.0, edge_corr_nm=60.0, wavy_nm=40.0, wavy_um=1.2, defects=0.0)
_NEW = dict(edge_nm=3.0, edge_corr_nm=80.0, wavy_nm=6.0, wavy_um=3.0, defects=0.0)


def build_structure(number: str) -> Structure:
    """The structure of product *number* (see :data:`PRODUCTS`)."""
    P2160 = 1.0 / 2.160
    au = dict(metal=MaterialId.GOLD)  # Au/Pd: gold stands in for the alloy
    if number in ("607", "607-A", "603", "603-A"):
        g = WaffleGrating(P2160, depth_nm=20.0, line_fraction=0.18, ramp_nm=45.0, **(_OLD if number.endswith("A") else _NEW))
        sph = LatexSpheres(261.0, 0.15) if number.startswith("603") else None
        return ShadowedReplicaStructure(g, rough_nm=7.0 if number.endswith("A") else 4.5, rough_um=0.1, spheres=sph, **au)
    if number == "606":
        return ShadowedReplicaStructure(WaffleGrating(P2160, depth_nm=30.0, line_fraction=0.3, both=False, **_NEW), **au)
    if number in ("677", "673"):
        g = WaffleGrating(0.5, depth_nm=40.0, line_fraction=0.5, ramp_nm=12.0, trench=True, **_NEW)
        sph = LatexSpheres(261.0, 0.15) if number == "673" else None
        return ShadowedReplicaStructure(g, rough_nm=2.0, spheres=sph, **au)
    if number == "628-B":
        return ShadowedReplicaStructure(None, base_nm=15.0, rough_nm=1.5, crumple_nm=4.0, metal_nm=15.0, elevation_deg=25.0,
                                        spheres=LatexSpheres(204.0, 0.8), island_nm=10.0, coverage=0.7, **au)
    if number.startswith("610-"):
        d = float(number.split("-")[1])
        return ShadowedReplicaStructure(None, base_nm=15.0, rough_nm=0.0, crumple_nm=0.0, metal_nm=0.0,
                                        spheres=LatexSpheres(d, min(20.0, 0.15 / (d / 1000.0) ** 2), 0.03))
    raise KeyError(f"no structure for product {number!r}")


def mean_thickness_nm(number: str) -> float:
    """Area-mean carbon-equivalent thickness of product *number* (its low-magnification look)."""
    s = build_structure(number)
    t = s.mean_carbon_nm() if isinstance(s, ShadowedReplicaStructure) else 20.0
    if isinstance(s, ShadowedReplicaStructure) and s.metal_nm > 0:
        t += s.flat_deposit_nm() * carbon_equivalent(s.metal)
    return t


def preset_name(number: str) -> str:
    return f"Ted Pella {number} - {PRODUCTS[number].name}"
