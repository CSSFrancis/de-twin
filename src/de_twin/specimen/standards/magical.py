"""MAG*I*CAL (675): a cross-section of silicon with four sets of Si(0.8)Ge(0.2) marker layers,
calibrated by Ted Pella, for magnification (1,000x to 1,000,000x with the Si(111) lattice),
camera constant and image / diffraction rotation.

Calibrated values (Ted Pella's datasheet): the sets are 1.25, 1.22, 1.18 and 1.14 um apart
(from the top, 5.21 um in all) and each spans 108.5, 107.4, 104.9 and 99.0 nm (five SiGe
layers of ~10-12 nm with ~12-14 nm of Si between them); Si <111> spacing 0.3135428 nm; the
cross-section is viewed down <011>.

Modelled: the sample surface (the glue line) at local y = 0 with the silicon towards +y,
the ion-milled perforation beyond the glue, a thickness wedge growing away from it, the
layers normal to Si [100] (the growth direction) lying along local x.
"""

from __future__ import annotations

import math

import numpy as np

from ...hashing import SeedKind, hash_seed
from ..geometry import NM_PER_UM
from ..materials import MaterialId, absorption_lengths_nm
from ..noise import fbm
from ..structures import _claim, _owner_pixels, _resolved
from .common import StandardStructure, angle_grain, angle_overrides

#: Distances (um) from the top to each marker set, cumulative, and each set's span (nm).
SET_SPACINGS_UM = (1.25, 1.22, 1.18, 1.14)
SET_SPANS_NM = (108.5, 107.4, 104.9, 99.0)
LAYERS_PER_SET = 5
SPACER_RATIO = 1.25  # Si spacer / SiGe layer thickness (span = 5 d + 4 x 1.25 d = 10 d)
SI_111_NM = 0.3135428


def layer_bands() -> list[tuple[float, float]]:
    """(top, bottom) um of every SiGe layer, below the surface."""
    out = []
    y = 0.0
    for gap, span in zip(SET_SPACINGS_UM, SET_SPANS_NM):
        y += gap
        d = span / (LAYERS_PER_SET + SPACER_RATIO * (LAYERS_PER_SET - 1)) / NM_PER_UM
        top = y - 0.5 * span / NM_PER_UM
        for k in range(LAYERS_PER_SET):
            y0 = top + k * d * (1.0 + SPACER_RATIO)
            out.append((y0, y0 + d))
    return out


class MagicalStructure(StandardStructure):
    base_material = MaterialId.SILICON

    def __init__(self, glue_um: float = 1.5, t0_nm: float = 40.0, wedge_nm_per_um: float = 25.0,
                 max_nm: float = 400.0):
        self.glue, self.t0, self.wedge, self.tmax = float(glue_um), float(t0_nm), float(wedge_nm_per_um), float(max_nm)
        self.bands = np.array(layer_bands())

    def base_nm(self) -> float:
        return 200.0

    def mean_nm(self) -> float:
        lam = absorption_lengths_nm(200.0)
        return 200.0 * float(lam[int(MaterialId.AMORPHOUS_CARBON)] / lam[int(MaterialId.SILICON)])

    def grain_textures(self):
        o = angle_overrides(MaterialId.SILICON, (0, 1, 1), (1, 0, 0))
        return (), o + angle_overrides(MaterialId.SILICON_GERMANIUM, (0, 1, 1), (1, 0, 0))

    def fill(self, ctx, owner):
        if not _resolved(0.2, ctx):
            return
        seed = int(hash_seed(owner.seed, SeedKind.STRUCTURE, 0)) & 0xFFFFFFFF
        base = self.base_nm()
        growth = owner.rot + 0.5 * math.pi  # Si [100] along local +y
        for B in _owner_pixels(ctx, owner):
            y = B.ly
            # the perforation's edge is ragged
            edge = -self.glue - 0.25 * (1.0 + fbm(seed ^ 0x71, B.lx, 0.0 * y, 0.8, 3))
            t = np.clip(self.t0 + self.wedge * (y - edge), 0.0, self.tmax)
            hole = y < edge
            glue = ~hole & (y < 0.0)
            si = y >= 0.0
            ctx.add_thickness(B.flat, np.where(hole, 0.0, t) - base)
            if hole.any():
                f = B.sub(hole)
                ctx.material[f] = MaterialId.VACUUM
                ctx.grain[f] = -1
            if glue.any():
                _claim(ctx, B.sub(glue), np.full(int(glue.sum()), -1, np.int32), MaterialId.AMORPHOUS_CARBON)
            if si.any():
                ys = y[si]
                sige = np.zeros(ys.shape, bool)
                for y0, y1 in self.bands:
                    sige |= (ys >= y0) & (ys < y1)
                mat = np.where(sige, MaterialId.SILICON_GERMANIUM, MaterialId.SILICON).astype(np.uint8)
                g = np.where(sige, angle_grain(MaterialId.SILICON_GERMANIUM, growth),
                             angle_grain(MaterialId.SILICON, growth)).astype(np.int32)
                _claim(ctx, B.sub(si), g, mat)
