"""The :class:`Specimen` facade: scene + clock + in-situ thermal budget."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Optional

import numpy as np

from ..state import HolderState
from .fieldmap import OPTIONAL_LAYERS, FieldMap, GrainTable, ViewWindow
from .materials import MaterialId
from .options import CLEARS_POPULATION, NON_REGENERATING, SpecimenConfig
from .raster import RasterContext, rasterize_scene
from .scene import Feature, Scene

# Thermal budget model (OpticsState.h / AdvanceThermalBudget).
REFERENCE_OVERHEAT_C = 100.0  # degrees above onset at which the budget advances at 1 s/s
MELT_RATE_PER_S = 40.0  # budget units shed per second at/above the melting point


def advance_thermal_budget(budget_s: float, temperature_c: float, onset_c: float, melt_c: float,
                           dt_s: float) -> float:
    """Integrate the crystallisation budget over ``dt_s`` at a constant temperature.

    Above the melting point the budget drains at 40 /s (re-amorphisation); below the onset it is
    frozen (a quench keeps whatever state it has); in between it grows linearly with overheat,
    1 s of budget per second at onset + 100 C.
    """
    if not dt_s > 0:
        return budget_s
    if temperature_c >= melt_c:
        return max(0.0, budget_s - MELT_RATE_PER_S * dt_s)
    if temperature_c < onset_c:
        return budget_s
    return budget_s + (temperature_c - onset_c) / REFERENCE_OVERHEAT_C * dt_s


def crystalline_fraction(nucleation_u, budget_s: float, start_s: float, end_s: float, growth_s: float):
    """Per-grain crystalline fraction (CrystalBinFor, continuous instead of 8 bins).

    Nucleation is log-uniform over the ``[start, end]`` budget window (more grains go early than
    late), followed by a linear growth ramp of ``growth_s``.
    """
    u = np.clip(np.asarray(nucleation_u, np.float64), 0.0, 1.0)
    tmin = max(0.1, start_s)
    tmax = max(tmin * 1.01, end_s)
    t_nuc = tmin * np.power(tmax / tmin, u)
    if not growth_s > 0:
        return (budget_s > t_nuc).astype(np.float32)
    return np.clip((budget_s - t_nuc) / growth_s, 0.0, 1.0).astype(np.float32)


@functools.lru_cache(maxsize=8)
def _grain_table(seed: int, textures: tuple = (), overrides: tuple = ()) -> GrainTable:
    return GrainTable.generate(seed, textures, overrides)


def grain_textures(scene: Scene) -> tuple[tuple, tuple]:
    """Default textures of a scene: ``<111>`` fibre for deposited thin films, a single crystal
    near a low-index zone ([110] for Si, [001] otherwise) for every FIB lamella matrix, and
    uniform random orientations for everything else (particles, polycrystals)."""
    from ..crystal.orientation import FIBRE_111, SI_110_LAMELLA, Texture
    from .preparations import BulkSamplePreparation, ThinFilmPreparation

    textures = []
    if isinstance(scene.preparation, ThinFilmPreparation):
        textures.append((int(scene.preparation.material), FIBRE_111))
    overrides = {}
    for area in scene.holder.areas:
        prep = scene.preparation_for(area)
        if isinstance(prep, BulkSamplePreparation):
            gid = prep._matrix_grain(area, scene.area_seed(area.index))
            if gid >= 0:
                overrides[int(gid)] = (SI_110_LAMELLA if prep.material == MaterialId.SILICON
                                       else Texture("single", (0, 0, 1), 0.6))
    return tuple(textures), tuple(sorted(overrides.items()))


class Specimen:
    """A deterministic virtual specimen: holder + preparation + time evolution.

    >>> s = Specimen(SpecimenConfig(seed=1))
    >>> fm = s.rasterize(ViewWindow(center_um=(0.0, 0.0), pixel_um=0.002, shape=(256, 256)))
    """

    def __init__(self, config: Optional[SpecimenConfig] = None):
        self.config = (config or SpecimenConfig()).copy()
        self.options = self.config.resolved_options()
        self._build()
        self._time_s = 0.0
        self._last_update_s: Optional[float] = None
        self._budget_s = 0.0
        self._temperature_c = self.options.specimen_temperature_c
        self._generation = 0

    def _build(self):
        self.scene = Scene(self.config, self.options)
        self.grains: GrainTable = _grain_table(int(self.config.seed) & 0xFFFFFFFF, *grain_textures(self.scene))

    # -- options --------------------------------------------------------------------------------
    def set_options(self, **kw) -> None:
        """Change options live. World-changing options regenerate the scene (and reset the
        thermal budget, as a new specimen is a new experiment); others apply in place."""
        new = self.options.replace(**kw)
        changed = {k for k in kw if getattr(new, k) != getattr(self.options, k)}
        self.config.options.update(kw)
        self.options = new
        if changed - NON_REGENERATING:
            self._build()
            self._budget_s = 0.0
        elif changed & CLEARS_POPULATION:
            self._build()  # baked into populated areas: rebuild, keep clock and thermal budget
        else:
            self.scene.opts = new

    # -- clock ----------------------------------------------------------------------------------
    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def time_index(self) -> int:
        return int(math.floor(self._time_s / self.options.time_step_s + 1e-9)) if self._time_s > 0 else 0

    @property
    def thermal_budget_s(self) -> float:
        return self._budget_s

    @property
    def temperature_c(self) -> float:
        return self._temperature_c

    @property
    def crystallization_enabled(self) -> bool:
        v = self.options.in_situ_crystallization
        return (self.config.holder == "insitu_heating_chip") if v is None else bool(v)

    @property
    def time_dependent(self) -> bool:
        return self.scene.time_dependent

    def update(self, time_s: float, holder: Optional[HolderState] = None) -> None:
        """Advance the scene clock to ``time_s`` and integrate the thermal budget.

        The temperature is ``holder.temperature_c`` when a holder state is given (and is not a
        ``kind == "none"`` placeholder), else ``options.specimen_temperature_c``. It is taken as
        constant over the interval since the previous update. The first call only sets the
        reference time. Time going backwards moves the clock but never un-crystallises.
        """
        time_s = float(time_s)
        if holder is not None and getattr(holder, "kind", "none") != "none":
            temp = float(holder.temperature_c)
        else:
            temp = float(self.options.specimen_temperature_c)
        if self._last_update_s is not None:
            dt = time_s - self._last_update_s
            self._budget_s = advance_thermal_budget(self._budget_s, temp, self.options.crystallization_onset_c,
                                                    self.options.melting_temperature_c, dt)
        self._last_update_s = time_s
        self._time_s = time_s
        self._temperature_c = temp

    def reset_clock(self) -> None:
        self._time_s = 0.0
        self._last_update_s = None
        self._budget_s = 0.0

    def crystallinity(self, grain_id) -> np.ndarray:
        """Crystalline fraction in [0, 1] per grain id.

        Grain -1 on crystalline material means "not yet nucleated" (glassy, 0) in a
        time-evolving in-situ scene, and "unresolved fine grains" (powder, 1) elsewhere.
        """
        g = np.asarray(grain_id)
        out = np.ones(g.shape, np.float32)
        if self.time_dependent:
            out[g < 0] = 0.0
        if not self.crystallization_enabled:
            return out
        ok = (g >= 0) & (g < self.grains.nucleation_u.shape[0])
        o = self.options
        out[ok] = crystalline_fraction(self.grains.nucleation_u[g[ok]], self._budget_s, o.nucleation_start_s,
                                       o.nucleation_end_s, o.crystal_growth_s)
        return out

    # -- geometry -------------------------------------------------------------------------------
    def bounds_um(self) -> tuple[float, float, float, float]:
        return self.scene.holder.bounds.as_tuple()

    def required_layers(self) -> frozenset:
        """Optional FieldMap layers this specimen can fill (descan/strain for FIB lamellae)."""
        return frozenset(self.scene.required_layers)

    def rasterize(self, view: ViewWindow, layers: frozenset = frozenset()) -> FieldMap:
        layers = frozenset(layers) & OPTIONAL_LAYERS
        drifted = view
        dx, dy = self.drift_um()
        if dx or dy:  # the specimen moved by (dx, dy): look at world - drift
            drifted = dataclasses.replace(view, center_um=(view.center_um[0] - dx, view.center_um[1] - dy))
        ctx = RasterContext(drifted, layers, grains=self.grains, time_index=self.time_index,
                            lod_threshold_px=self.options.lod_threshold_px)
        rasterize_scene(self.scene, ctx, curtain_depth=min(max(self.options.curtain_depth, 0.0), 1.0))
        self._generation += 1
        self.last_stats = dict(ctx.stats, **self.scene.last_stats)
        fm = ctx.to_fieldmap(time_s=self._time_s, generation=self._generation)
        fm.view = view
        return fm

    def drift_um(self) -> tuple[float, float]:
        """Accumulated specimen drift (um) at the current time index."""
        t = self.time_index if self.time_dependent else 0
        o = self.options
        return (o.drift_per_step_x_nm * t / 1000.0, o.drift_per_step_y_nm * t / 1000.0)

    def features(self) -> list[Feature]:
        return self.scene.features()

    def nearest_feature(self, x_um: float, y_um: float) -> Optional[Feature]:
        return self.scene.nearest_feature(float(x_um), float(y_um))

    def overview(self, shape=(1024, 1024)) -> np.ndarray:
        """Whole-holder projected-thickness image (DebugOverview), float32 nm, 5 % padding."""
        xmin, ymin, xmax, ymax = self.bounds_um()
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        ny, nx = int(shape[0]), int(shape[1])
        px = max(1.05 * (xmax - xmin) / nx, 1.05 * (ymax - ymin) / ny)
        fm = self.rasterize(ViewWindow(center_um=(cx, cy), pixel_um=px, shape=(ny, nx)))
        return fm.thickness_nm
