"""What the calibration standards share: the structure interface, a perforated (holey) carbon
film, objects laid out cell by cell, and grain orientations tied to an in-plane angle."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from ...hashing import SeedKind, hash_seed, normal_from_hash, splitmix64, uniform_from_hash
from ..fieldmap import GRAINS_PER_MATERIAL
from ..geometry import NM_PER_UM
from ..materials import MaterialId
from ..noise import fbm
from ..structures import Structure

U64 = np.uint64


class StandardStructure(Structure):
    """A standard's structure. Its owner is a rectangle of `base_material`, `base_nm` thick,
    that the structure edits; `mean_nm` is the carbon-equivalent area mean (the aggregate at
    low magnification) and `grain_textures` fixes crystal orientations it relies on."""

    required_layers = frozenset()
    base_material: int = MaterialId.AMORPHOUS_CARBON

    def base_nm(self) -> float:
        raise NotImplementedError

    def mean_nm(self) -> float:
        return self.base_nm()

    def grain_textures(self) -> tuple[tuple, tuple]:
        return (), ()


# ------------------------------------------------------------------ cells of objects
def cells_near(lx, ly, cell_um: float, reach_um: float) -> Iterator[tuple[int, int]]:
    """Lattice cells (i, j) whose objects (reaching `reach_um` from their cell) can touch the
    pixels at owner-local (lx, ly)."""
    x0, x1 = float(np.min(lx)) - reach_um, float(np.max(lx)) + reach_um
    y0, y1 = float(np.min(ly)) - reach_um, float(np.max(ly)) + reach_um
    for j in range(math.floor(y0 / cell_um), math.floor(y1 / cell_um) + 1):
        for i in range(math.floor(x0 / cell_um), math.floor(x1 / cell_um) + 1):
            yield i, j


def cell_rng(seed: int, salt: int, i: int, j: int) -> "CellRng":
    return CellRng(int(hash_seed(seed, SeedKind.STRUCTURE, (i * 73856093) ^ (j * 19349663), salt)))


class CellRng:
    """A tiny deterministic stream of numbers from a cell's hash."""

    def __init__(self, h: int):
        self.h = int(h) & 0xFFFFFFFFFFFFFFFF

    def _next(self) -> int:
        self.h = int(splitmix64(self.h))
        return self.h

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * float(uniform_from_hash(self._next()))

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return mu + sigma * float(normal_from_hash(self._next()))

    def lognormal(self, median: float, sigma: float) -> float:
        return median * math.exp(sigma * float(normal_from_hash(self._next())))

    def poisson(self, lam: float) -> int:
        """Knuth's method (small lam)."""
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.uniform()
            if p <= L:
                return k
            k += 1

    def hash(self) -> int:
        return self._next()


# ------------------------------------------------------------------ orientation buckets
#: In-plane orientations are quantised to this many buckets over 180 degrees (0.35 deg).
ANGLE_BUCKETS = GRAINS_PER_MATERIAL


def angle_bucket(angle_rad) -> np.ndarray:
    """The bucket of an in-plane angle (radians, modulo pi)."""
    return np.mod(np.rint(np.asarray(angle_rad) / math.pi * ANGLE_BUCKETS).astype(np.int64), ANGLE_BUCKETS)


def angle_grain(material: int, angle_rad) -> np.ndarray:
    """Grain id of `material` whose orientation is the bucket of `angle_rad` (see
    `angle_overrides`)."""
    return (int(material) * GRAINS_PER_MATERIAL + angle_bucket(angle_rad)).astype(np.int32)


def angle_overrides(material: int, axis, inplane, spread_deg: float = 0.0) -> tuple:
    """Grain overrides giving grain k of `material` zone axis `axis` along the beam and crystal
    direction `inplane` at 180 k / ANGLE_BUCKETS degrees: crystals whose shape fixes their
    orientation pick their grain with `angle_grain`."""
    from ...crystal.orientation import Texture

    base = int(material) * GRAINS_PER_MATERIAL
    return tuple((base + k, Texture("oriented", tuple(axis), spread_deg, tuple(inplane), 180.0 * k / ANGLE_BUCKETS))
                 for k in range(ANGLE_BUCKETS))


# ------------------------------------------------------------------ holey carbon
class HoleyFilm:
    """A perforated carbon film (Ted Pella 609 and the support of several standards): holes
    from breath-figure casting, round but not perfectly, log-normal in size, crowded (big,
    medium and small populations that overlap and merge). Some big holes keep a thin torn
    membrane across them, itself full of small holes."""

    def __init__(self, thickness_nm: float = 15.0, big_cell_um: float = 0.8, big_median_um: float = 0.3,
                 big_fraction: float = 1.0, small_cell_um: float = 0.28, small_median_um: float = 0.08,
                 small_fraction: float = 0.9, sigma: float = 0.4, wobble: float = 0.06,
                 tiny_cell_um: float = 0.15, tiny_median_um: float = 0.03, tiny_fraction: float = 0.3,
                 membrane_fraction: float = 0.15, membrane_nm: float = 4.0):
        self.t = float(thickness_nm)
        self.pop = ((big_cell_um, big_median_um, big_fraction, 0xB1), (small_cell_um, small_median_um, small_fraction, 0xB2),
                    (tiny_cell_um, tiny_median_um, tiny_fraction, 0xB3))
        self.sigma, self.wobble = float(sigma), float(wobble)
        self.membrane_fraction, self.membrane_nm = float(membrane_fraction), float(membrane_nm)

    def hole_fraction(self) -> float:
        f = 0.0
        for cell, med, frac, _ in self.pop:
            r2 = med * med * math.exp(2.0 * self.sigma ** 2)
            f += frac * math.pi * min(r2, (0.5 * cell) ** 2) / cell ** 2
        return min(f, 0.9)

    def in_hole(self, seed: int, lx, ly) -> np.ndarray:
        """Whether owner-local (lx, ly) um is in a hole (membranes ignored)."""
        return self._holes(seed, lx, ly)[0]

    def thickness(self, seed: int, lx, ly) -> np.ndarray:
        """Film thickness (nm) at owner-local (lx, ly) um: `t`, 0 in a hole, a thin perforated
        membrane across some big holes."""
        hole, membrane = self._holes(seed, lx, ly)
        t = np.where(hole, 0.0, self.t)
        if membrane.any():
            # a torn membrane: the high part of a smooth field is left, lacy holes in it
            n = fbm(seed ^ 0xB9, lx[membrane], ly[membrane], 0.06, 3, 0.5)
            t[membrane] = np.where(n > -0.05, self.membrane_nm * (1.0 + 0.3 * n), 0.0)
        return t

    def _holes(self, seed: int, lx, ly):
        from ..geometry import JitteredLattice

        out = np.zeros(np.shape(lx), bool)
        membrane = np.zeros(np.shape(lx), bool)
        for cell, med, frac, salt in self.pop:
            lat = JitteredLattice(cell, seed, salt, jitter_frac=0.5)
            hit = lat.nearest(lx, ly)
            h = hit.h
            present = uniform_from_hash(h ^ U64(0x5A)) < frac
            r = med * np.exp(self.sigma * normal_from_hash(h ^ U64(0x5B)))
            r = np.minimum(r, 0.45 * cell)  # a hole stays inside its cell
            ang = np.arctan2(ly - hit.sy, lx - hit.sx)
            # a gently irregular rim: low harmonics of the angle
            ph1 = 2 * math.pi * uniform_from_hash(h ^ U64(0x5C))
            ph2 = 2 * math.pi * uniform_from_hash(h ^ U64(0x5D))
            rr = r * (1.0 + self.wobble * (np.cos(2 * ang + ph1) + 0.6 * np.cos(3 * ang + ph2)))
            h_in = present & (hit.d2 < rr * rr)
            out |= h_in
            if salt == 0xB1 and self.membrane_fraction > 0:
                membrane |= h_in & (uniform_from_hash(h ^ U64(0x5E)) < self.membrane_fraction)
        return out, membrane & out


def sphere_chord_nm(d2_um2, r_um) -> np.ndarray:
    """Projected thickness (nm) of a sphere of radius `r_um` at squared distance `d2_um2`."""
    return 2.0 * np.sqrt(np.maximum(r_um * r_um - d2_um2, 0.0)) * NM_PER_UM


def rough(seed: int, lx, ly, amp_nm: float, scale_nm: float) -> np.ndarray:
    """A small thickness roughness (nm) on world-locked noise."""
    return amp_nm * fbm(seed, lx, ly, scale_nm / NM_PER_UM, 3)
