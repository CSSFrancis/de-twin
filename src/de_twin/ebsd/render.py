"""EBSD rendering: per scan point, project the grain's master pattern onto the detector.

Per scan point (one per raster pixel of the EBSD optics' view, pitch = scan step on the
tilt-corrected sample surface):

1. the microstructure under the beam (:meth:`de_twin.specimen.bulk.BulkSurface.sample`): grain,
   phase, orientation, the neighbouring grain and the distance to the boundary between them,
   and the surface quality;
2. near a boundary the neighbour's pattern is mixed in with the fraction of the interaction
   volume's footprint (an ellipse ~0.02 x the Kanaya-Okayama range across the tilt axis,
   1 / cos(tilt) longer along it) that lies across the boundary;
3. the pattern centre for the beam position (:meth:`EbsdGeometry.pattern_centre`);
4. one numba pass (:func:`de_twin.ebsd.kernels.project`) over the binned detector grid:
   gnomonic direction of each pixel, master-pattern lookup in the crystal frame, the
   backscatter background (cosine + forward lobe, x the backscatter yield and the beam
   current), the solid angle of the pixel, and the band modulation
   ``1 + band_contrast * quality * m``.

The per-scan tables (step 1-2) are built once per scan (view) and kept; master patterns are
process-wide (:mod:`de_twin.ebsd.master`). Patterns are rendered at the binned detector shape
directly (no full-sensor render), in electrons per *sensor* pixel per second (the detector's
flux unit, so ``Detector.expose`` takes them as they are).

A specimen without :meth:`sample` (a TEM specimen) is read from its field map: phase and
grain-table orientation per scan point, no boundary mixing.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..specimen.materials import GRAINS_PER_MATERIAL, material
from . import kernels
from .geometry import (
    EbsdGeometry,
    background_parameters,
    backscatter_yield,
    boundary_mix,
    footprint_um,
    pc_pixels,
)
from .master import master_for

ELECTRON_C = 1.602176634e-19
_EMPTY_MASTER = np.zeros((3, 3), np.uint8)


@dataclass
class EbsdTables:
    """Per scan point (ny, nx) microstructure of one scan."""

    x_um: np.ndarray  # beam position on the sample (sample frame), float64
    y_um: np.ndarray
    material: np.ndarray  # uint8, 0 = no crystal
    material2: np.ndarray
    grain_id: np.ndarray  # int32, -1 = none
    grain_id2: np.ndarray
    orient: np.ndarray  # (ny, nx) int32 index into matrices (-1 = none)
    orient2: np.ndarray
    matrices: np.ndarray  # (K, 3, 3) crystal -> sample
    mix: np.ndarray  # float32 weight of the neighbour (0 .. 0.5)
    quality: np.ndarray  # float32 surface quality (0, 1]

    @property
    def nbytes(self) -> int:
        return int(sum(getattr(self, k).nbytes for k in self.__dataclass_fields__))


def ebsd_geometry(optics) -> EbsdGeometry:
    g = optics.extras.get("ebsd")
    return g if isinstance(g, EbsdGeometry) else EbsdGeometry()


def _key(optics) -> tuple:
    g = ebsd_geometry(optics)
    return (optics.view, round(optics.ht_kv, 4), g.sample_tilt_deg, g.boundary_blend, g.interaction_scale)


class EbsdRenderer:
    """Owns the EBSD scan tables; renders patterns, pattern stacks and ground truth."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._tables: "OrderedDict[tuple, EbsdTables]" = OrderedDict()
        self._lock = threading.RLock()
        self.patterns_rendered = 0

    # ------------------------------------------------------------------ tables
    def tables(self, specimen, optics, field_map=None, grains=None) -> EbsdTables:
        key = _key(optics)
        with self._lock:
            t = self._tables.get(key)
            if t is not None:
                self._tables.move_to_end(key)
                return t
        t = self._build(specimen, optics, field_map, grains)
        with self._lock:
            self._tables[key] = t
            while len(self._tables) > 2:
                self._tables.popitem(last=False)
        return t

    def _build(self, specimen, optics, field_map, grains) -> EbsdTables:
        g = ebsd_geometry(optics)
        view = optics.view
        ny, nx = view.shape
        X, Y = view.world_grid()
        x, y = np.ascontiguousarray(X, np.float64), np.ascontiguousarray(Y, np.float64)
        if callable(getattr(specimen, "sample", None)):
            s = specimen.sample(x.ravel(), y.ravel(), blend=True)
            keys = np.concatenate([s["hash"], s["hash2"]])
            phases = np.concatenate([s["phase"], s["phase2"]])
            uk, first, inv = np.unique(keys, return_index=True, return_inverse=True)
            mats = specimen.orientations(uk, phases[first])
            n = x.size
            o1, o2 = inv[:n].astype(np.int32), inv[n:].astype(np.int32)
            mix = np.zeros(n, np.float32)
            if g.boundary_blend:
                m1 = s["material"]
                for mid in np.unique(m1):
                    sel = m1 == mid
                    mm = material(int(mid))
                    rx, ry = footprint_um(mm.z_eff, mm.density_g_cm3, optics.ht_kv, g.sample_tilt_deg,
                                          g.interaction_scale)
                    mix[sel] = boundary_mix(s["boundary_um"][sel], s["normal"][sel], rx, ry)
            mix[o1 == o2] = 0.0
            return EbsdTables(x, y, s["material"].reshape(ny, nx), s["material2"].reshape(ny, nx),
                              s["grain_id"].reshape(ny, nx), s["grain_id2"].reshape(ny, nx),
                              o1.reshape(ny, nx), o2.reshape(ny, nx), mats, mix.reshape(ny, nx),
                              s["quality"].reshape(ny, nx))
        # a TEM specimen: the field map's phase and grain-table orientation
        fm = field_map
        gid = fm.grain_id.astype(np.int64)
        mat = fm.material_id.copy()
        crystalline = np.array([material(int(m)).crystal is not None for m in range(int(mat.max()) + 1)])
        ok = (gid >= 0) & (gid // GRAINS_PER_MATERIAL == mat) & crystalline[mat]
        if grains is not None:
            ok &= gid < len(grains)
        uniq, inv = np.unique(gid[ok], return_inverse=True)
        mats = grains.matrices[uniq] if (grains is not None and len(uniq)) else np.zeros((0, 3, 3))
        orient = np.full(gid.shape, -1, np.int32)
        orient[ok] = inv
        mat = np.where(ok, mat, 0).astype(np.uint8)
        gid32 = np.where(ok, gid, -1).astype(np.int32)
        return EbsdTables(x, y, mat, mat.copy(), gid32, gid32.copy(), orient, orient.copy(), mats,
                          np.zeros((ny, nx), np.float32), np.ones((ny, nx), np.float32))

    # ------------------------------------------------------------------ geometry per point
    def pattern_centre(self, optics, tab: EbsdTables, ix: int, iy: int) -> np.ndarray:
        g = ebsd_geometry(optics)
        cx, cy = optics.view.center_um
        return g.pattern_centre(tab.x_um[iy, ix] - cx, tab.y_um[iy, ix] - cy, optics.extras["sensor_shape"],
                                optics.extras["sensor_pixel_um"])

    def pattern_centres(self, optics, tab: EbsdTables) -> np.ndarray:
        g = ebsd_geometry(optics)
        cx, cy = optics.view.center_um
        return g.pattern_centre(tab.x_um - cx, tab.y_um - cy, optics.extras["sensor_shape"],
                                optics.extras["sensor_pixel_um"])

    @staticmethod
    def grid(optics) -> tuple[int, int, int, int, int, int]:
        """(h, w, ox, oy, bx, by): the binned detector grid and its sensor offset."""
        bx, by = optics.extras.get("hw_binning", (1, 1))
        oh, ow = optics.output_shape
        ox, oy = optics.extras.get("roi_origin", (0, 0))
        return max(1, oh // by), max(1, ow // bx), int(ox), int(oy), int(bx), int(by)

    # ------------------------------------------------------------------ one pattern
    def pattern(self, optics, matrix, pc, *, material_id: int, matrix2=None, material2: Optional[int] = None,
                mix: float = 0.0, quality: float = 1.0, raw: bool = False, out=None) -> np.ndarray:
        """One pattern on the optics' binned detector grid for a grain with crystal->sample
        ``matrix`` at Bruker pattern centre ``pc``. ``raw``: the projected (standardised)
        master values only. Otherwise electrons per sensor pixel per second."""
        g = ebsd_geometry(optics)
        h, w, ox, oy, bx, by = self.grid(optics)
        out = np.empty((h, w), np.float32) if out is None else out
        x0, y0, dd = pc_pixels(pc, optics.extras["sensor_shape"])
        s2d = g.sample_to_detector()
        if material_id and matrix is not None:
            m1 = np.ascontiguousarray((s2d @ matrix).T)
            mp = master_for(material_id, optics.ht_kv, g)
            mp1, a1, b1 = mp.data, mp.offset, mp.scale
        else:
            m1, mp1, a1, b1 = np.eye(3), _EMPTY_MASTER, 0.0, 0.0
        w2 = float(mix) if (matrix2 is not None and material2) else 0.0
        if w2 > 0:
            m2 = np.ascontiguousarray((s2d @ matrix2).T)
            mp = master_for(material2, optics.ht_kv, g)
            mp2, a2, b2 = mp.data, mp.offset, mp.scale
        else:
            m2, mp2, a2, b2 = m1, mp1, a1, b1
        eta = backscatter_yield(material(material_id).z_eff, g.sample_tilt_deg) if material_id else 0.0
        e_s = float(optics.pattern_e_per_s) * eta
        lobe, kappa = background_parameters(optics.ht_kv, g.sample_tilt_deg)
        if not g.background:  # isotropic: e_s / 4 pi per steradian (x the pixel's solid angle)
            lobe, kappa = 1.0, 0.0
        contrast = float(g.band_contrast) * float(quality) if material_id else 0.0
        kernels.project(mp1, float(a1), float(b1), m1, mp2, float(a2), float(b2), m2, w2, float(x0),
                        float(y0), float(dd), float(ox), float(oy), float(bx),
                        float(by), bool(raw), g.surface_normal_d(), g.forward_direction_d(),
                        e_s * (1.0 - lobe), e_s * lobe, float(kappa), contrast, out)
        self.patterns_rendered += 1
        return out

    def render(self, specimen, optics, ix: int, iy: int, field_map=None, grains=None) -> np.ndarray:
        """The pattern at scan point (ix, iy): float32 (h, w) binned grid, electrons per sensor
        pixel per second."""
        tab = self.tables(specimen, optics, field_map, grains)
        return self._render_point(optics, tab, ix, iy)

    def _render_point(self, optics, tab: EbsdTables, ix: int, iy: int, raw: bool = False, out=None):
        o1, o2 = int(tab.orient[iy, ix]), int(tab.orient2[iy, ix])
        m = int(tab.material[iy, ix])
        mat1 = tab.matrices[o1] if o1 >= 0 else None
        mat2 = tab.matrices[o2] if o2 >= 0 else None
        return self.pattern(optics, mat1, self.pattern_centre(optics, tab, ix, iy),
                            material_id=m if o1 >= 0 else 0,
                            matrix2=mat2, material2=int(tab.material2[iy, ix]), mix=float(tab.mix[iy, ix]),
                            quality=float(tab.quality[iy, ix]), raw=raw, out=out)

    # ------------------------------------------------------------------ stacks
    def patterns(self, specimen, optics, field_map=None, grains=None, *, out=None, dtype=np.float32,
                 per_binned_pixel: bool = True) -> np.ndarray:
        """Every scan point's pattern: (ny, nx, h, w). ``per_binned_pixel``: electrons per binned
        pixel per second (the 4D-STEM datacube's unit); else per sensor pixel. ``out`` may be a
        preallocated (e.g. memory-mapped) array of that shape."""
        tab = self.tables(specimen, optics, field_map, grains)
        ny, nx = tab.material.shape
        h, w, _, _, bx, by = self.grid(optics)
        if out is None:
            out = np.empty((ny, nx, h, w), dtype)
        buf = np.empty((h, w), np.float32)
        k = np.float32(bx * by if per_binned_pixel else 1)
        for iy in range(ny):
            for ix in range(nx):
                self._render_point(optics, tab, ix, iy, out=buf)
                out[iy, ix] = buf * k if k != 1 else buf
        return out

    # ------------------------------------------------------------------ ground truth
    def ground_truth(self, specimen, optics, field_map=None, grains=None) -> dict:
        """Per scan point: ``material_id`` (= phase id; 0 none), ``grain_id`` (-1 none),
        ``quaternion`` (orix rotation of the grain, *sample* frame, a >= 0, identity where there
        is no crystal), ``pattern_centre`` (Bruker pcx, pcy, pcz), ``band_contrast`` (the
        dominant grain's band modulation, ``band_contrast * quality * (1 - mix)``),
        ``quality`` (surface quality x (1 - mix)), ``mix`` / ``grain_id2`` / ``material_id2``
        (the neighbour mixed in near a boundary), ``x_um`` / ``y_um`` (beam on the sample)."""
        from ..crystal import rotation_from_crystal_to_lab

        g = ebsd_geometry(optics)
        tab = self.tables(specimen, optics, field_map, grains)
        ny, nx = tab.material.shape
        q = np.zeros((ny, nx, 4))
        q[..., 0] = 1.0
        ok = tab.orient >= 0
        if ok.any():
            quats = rotation_from_crystal_to_lab(tab.matrices)
            q[ok] = quats[tab.orient[ok]]
        dominant = (1.0 - tab.mix).astype(np.float32)
        quality = np.where(ok, tab.quality * dominant, 0.0).astype(np.float32)
        return {
            "material_id": np.where(ok, tab.material, 0).astype(np.uint8),
            "grain_id": np.where(ok, tab.grain_id, -1).astype(np.int32),
            "quaternion": q,
            "pattern_centre": self.pattern_centres(optics, tab),
            "band_contrast": (quality * np.float32(g.band_contrast)).astype(np.float32),
            "quality": quality,
            "mix": tab.mix.copy(),
            "grain_id2": tab.grain_id2.copy(),
            "material_id2": tab.material2.copy(),
            "x_um": tab.x_um.copy(),
            "y_um": tab.y_um.copy(),
            "view": optics.view,
            "pc_convention": "bruker",
            "sample_tilt_deg": g.sample_tilt_deg,
            "detector_tilt_deg": g.detector_tilt_deg,
        }
