"""Calibration standards: Ted Pella's TEM / STEM calibration and test specimens.

Each product is a preset in :data:`~de_twin.specimen.CALIBRATION_STANDARDS`
(``"Ted Pella 607-A - ..."``), with what the catalogue publishes about it in :data:`PRODUCTS`
(``numbers``: the values a calibration is checked against) and what the twin had to assume in
``model``. Families:

* **Grating replicas** (607, 607-A, 606, 603, 603-A, 677, 673) and **gold-shadowed latex**
  (628-B): shadowed carbon replicas (:mod:`.replica`).
* **Latex spheres** (610-xx nominal, 610-5x..7x certified nanosphere standards).
* **Diffraction and lattice**: evaporated aluminium (619, camera length), molybdenum trioxide
  (625, image / diffraction rotation), oriented gold foil (646: 0.204, 0.144, 0.102 nm),
  graphitized carbon black (645: 0.34 nm), MAG*I*CAL (675: Si/SiGe cross-section).
* **Resolution**: catalase (612: 8.75 and 6.85 nm), ferritin (608: the tetrad, < 1.25 nm),
  gold on holey carbon (613), evaporated Pt/Ir (611), holey carbon (609, astigmatism) and the
  combined test specimen (638).

Not modelled: the NiOx AEM specimen (650, an X-ray / EELS test), the TEM CHECKER (602-15,
opaque Mn discs), the grid holder and the kits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..materials import MaterialId
from .biological import CatalaseStructure, FerritinStructure
from .common import HoleyFilm, StandardStructure
from .crystals import CarbonBlackParticles, MoO3LathsStructure
from .films import HoleyCarbonStructure, OrientedFoilStructure, PolycrystalFilmStructure
from .magical import SI_111_NM, MagicalStructure
from .replica import (LatexSpheres, ShadowedReplicaStructure, WaffleGrating, carbon_equivalent, fbm,
                      island_film)

__all__ = ["PRODUCTS", "Product", "build_structure", "mean_thickness_nm", "preset_name", "LatexSpheres",
           "ShadowedReplicaStructure", "WaffleGrating", "StandardStructure", "island_film", "fbm",
           "carbon_equivalent", "FAMILIES"]


@dataclass(frozen=True)
class Product:
    """A Ted Pella calibration standard: what the catalogue says it is (``numbers``: published
    values) and what the twin assumes where it says nothing (``model``)."""

    number: str
    name: str
    family: str
    description: str
    use: str
    numbers: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)


PRODUCTS: dict[str, Product] = {}

#: Families, in the order a menu shows them.
FAMILIES = ("Grating replicas", "Latex spheres", "Diffraction and lattice", "Resolution")


def _p(number, name, family, description, use, numbers=None, model=None):
    PRODUCTS[number] = Product(number, name, family, description, use, dict(numbers or {}), dict(model or {}))


G, L, D, R = FAMILIES
_REPLICA = "Carbon replica with Au/Pd shadowing"
_p("607", "2160 l/mm grating replica (waffle)", G,
   f"{_REPLICA} of a cross-line (waffle) diffraction grating, 2,160 lines/mm in both directions; "
   "G400 copper Gilder grid.", "Magnification calibration up to ~80-100,000x.",
   {"lines_per_mm": 2160.0, "period_nm": 462.9})
_p("607-A", "2160 l/mm grating replica, special applications (waffle)", G,
   f"{_REPLICA}, 2,160 lines/mm waffle; the older style, with more defects and a less clear pattern "
   "than the current 607; G400 copper grid.", "Aberration corrector alignment and SerialEM.",
   {"lines_per_mm": 2160.0, "period_nm": 462.9})
_p("606", "2160 l/mm grating replica (parallel lines)", G,
   f"{_REPLICA} of a parallel-line grating, 462.9 nm line spacing (2,160 lines/mm); G400 copper grid.",
   "Magnification calibration up to ~40-50,000x.", {"lines_per_mm": 2160.0, "period_nm": 462.9})
_p("603", "2160 l/mm grating replica with 0.261 um latex spheres", G,
   f"{_REPLICA}, 2,160 lines/mm, with 0.261 um latex spheres; G400 copper Gilder grid.",
   "Two-in-one magnification calibration (the spheres a double check).",
   {"lines_per_mm": 2160.0, "period_nm": 462.9, "sphere_nm": 261.0})
_p("603-A", "2160 l/mm grating replica with 0.261 um latex spheres, special applications", G,
   f"{_REPLICA}, the older-style 2,160 lines/mm grating with 0.261 um latex spheres; G400 copper grid.",
   "Aberration corrector alignment and SerialEM; magnification double check.",
   {"lines_per_mm": 2160.0, "period_nm": 462.9, "sphere_nm": 261.0})
_p("677", "500 nm cross-line grating replica", G,
   f"{_REPLICA} of a cross-line grating with trench-type grooves, 500 nm pitch (2,000 lines/mm); "
   "G400 copper grid.", "Magnification calibration up to 100,000x.", {"lines_per_mm": 2000.0, "period_nm": 500.0})
_p("673", "500 nm cross-line grating replica with 261 nm latex spheres", G,
   f"{_REPLICA}, 500 nm pitch (2,000 lines/mm), with 261 nm latex spheres; G400 copper grid.",
   "Two-in-one magnification calibration up to 150,000x.",
   {"lines_per_mm": 2000.0, "period_nm": 500.0, "sphere_nm": 261.0})
_p("628-B", "Gold-shadowed latex", G, "Latex particles, 0.204 um diameter, shadowed with gold; 3.05 mm grid.",
   "A test object for STEM.", {"sphere_nm": 204.0})
_GRATING_MODEL = {"replica_film_nm": 20.0, "relief_nm": 20.0, "metal": "gold (for Au/Pd)", "metal_nm": 6.0,
                  "shadow_elevation_deg": 25.0, "island_nm": 9.0}
for _n in ("607", "607-A", "606", "603", "603-A", "677", "673"):
    PRODUCTS[_n].model.update(_GRATING_MODEL)

# latex: (product, nominal um, uniformity %) and certified nanospheres (product, nominal nm,
# certified mean nm, +/- nm, standard deviation nm or None)
_NOMINAL = (("610-03", 0.03, 30), ("610-08", 0.08, 18), ("610-10", 0.09, 15), ("610-14", 0.17, 5),
            ("610-17", 0.26, 3), ("610-20", 0.30, 3), ("610-30", 0.49, 3), ("610-38", 1.00, 3))
_CERTIFIED = (("610-50", 20, 21.0, 1.5, None), ("610-53", 50, 46.0, 2.0, 7.2), ("610-56", 80, 81.0, 2.7, 5.8),
              ("610-58", 100, 97.0, 3.0, 4.5), ("610-60", 150, 151.0, 4.0, 3.2), ("610-61", 200, 200.0, 6.0, 3.4),
              ("610-63", 240, 240.0, 6.0, 3.7), ("610-66", 350, 350.0, 7.0, 4.7), ("610-69", 500, 491.0, 4.0, 6.3),
              ("610-73", 600, 596.0, 6.0, 7.7), ("610-76", 900, 903.0, 9.0, 9.3))
for _n, _um, _u in _NOMINAL:
    _p(_n, f"Latex spheres, {_um:g} um nominal", L,
       f"Polystyrene latex spheres, {_um:g} um nominal, size uniformity <= {_u}%, dried on a carbon film.",
       "Magnification test specimen.", {"sphere_nm": 1000.0 * _um, "uniformity_pct": float(_u)},
       {"cv": _u / 100.0, "film_nm": 15.0})
for _n, _nom, _mean, _pm, _sd in _CERTIFIED:
    _p(_n, f"Certified nanospheres, {_nom} nm", L,
       f"NIST-traceable polystyrene nanosphere size standard: certified mean {_mean:g} +/- {_pm:g} nm"
       + (f", standard deviation {_sd:g} nm" if _sd else "") + "; dried on a carbon film.",
       "Size / magnification calibration.",
       {"sphere_nm": _mean, "sphere_uncertainty_nm": _pm, **({"sphere_sd_nm": _sd} if _sd else {})},
       {"cv": (_sd or 0.05 * _mean) / _mean, "film_nm": 15.0})

_p("619", "Diffraction standard, evaporated aluminum", D,
   "Evaporated aluminum, ~31 nm thick; very small crystallites give ring patterns; G400 copper Gilder grid.",
   "Camera length calibration.", {"thickness_nm": 31.0, "a_nm": 0.40495},
   {"grain_nm": 15.0, "texture": "random"})
_p("625", "Image rotation standard, molybdenum trioxide", D,
   "Molybdenum trioxide crystals; G400 copper Gilder grid.",
   "Rotation between the selected-area image and its diffraction pattern.",
   {"lath_axis": "[001] (Pbnm) = [010] (Pnma), 0.3697 nm"},
   {"film_nm": 15.0, "lath_um": 1.5, "lath_thickness_nm": 40.0, "face": "(010) Pbnm"})
_p("646", "Oriented single-crystal gold foil", D,
   "An oriented single crystal of gold on a gold grid; lattice spacings 0.204, 0.144 and 0.102 nm.",
   "Resolution, image quality, magnification and stability of high-performance TEMs.",
   {"d_nm": (0.204, 0.144, 0.102)}, {"zone": "[001]", "thickness_nm": 15.0, "mosaic_deg": 0.4})
_p("645", "Graphitized carbon black", D,
   "Graphitized carbon black with 0.34 nm lattice plane spacing; standard 3 mm grid.",
   "Standard resolution test for TEMs.", {"d_nm": 0.34},
   {"particle_nm": "20-50, polyhedral, hollow", "support": "holey carbon"})
_p("675", "MAG*I*CAL", D,
   "Si cross-section with four Si/SiGe marker sets (calibrated spacings and widths), viewed down <011>.",
   "All TEM magnification ranges (1,000x to 1,000,000x, +/- 1.4%), camera constant, image / diffraction rotation.",
   {"si_111_nm": SI_111_NM, "set_spacings_um": (1.25, 1.22, 1.18, 1.14), "total_um": 5.21,
    "set_spans_nm": (108.5, 107.4, 104.9, 99.0), "accuracy_pct": 1.4, "zone": "<011>"},
   {"sige": "Si0.8Ge0.2", "glue_line": "local y = 0", "wedge_nm_per_um": 25.0})
_p("612", "Catalase crystals", R, "Negatively stained catalase crystals; 3 mm grid.",
   "High magnification calibration.", {"lattice_nm": (8.75, 6.85)},
   {"stain": "uranyl", "crystal_nm": 30.0})
_p("608", "Ferritin", R, "Ferritin molecules on formvar/carbon; G400 copper grid.",
   "Resolution: resolving the tetrad of the iron core indicates better than 1.25 nm.",
   {"tetrad_resolution_nm": 1.25}, {"shell_nm": 12.0, "core_nm": 7.0})
_p("613", "Gold on holey carbon", R, "Evaporated gold on holey carbon; G400 copper grid.",
   "Astigmatism correction and resolution at once, from the spaces between the gold islands.", {},
   {"island_nm": 12.0, "coverage": 0.72})
_p("611", "Evaporated platinum/iridium", R, "Evaporated Pt/Ir on perforated (holey) carbon film; 3 mm grid.",
   "Particle separation (resolution) test.", {}, {"metal": "platinum (for Pt/Ir)", "island_nm": 2.5})
_p("609", "Holey carbon film", R, "Thin carbon film with small holes; G400 copper grid.",
   "The best general astigmatism check (Fresnel fringe at the hole edges); stability.", {},
   {"film_nm": 15.0})
_p("638", "Combined test specimen", R,
   "Gold-shadowed holey carbon film with graphitized carbon particles; 3 mm grid.",
   "Assessment of what limits the microscope's performance; contamination.", {"d_nm": 0.34},
   {"gold": "islands, 8 nm"})
del G, L, D, R


def build_structure(number: str) -> StandardStructure:
    """The structure of product *number* (see :data:`PRODUCTS`)."""
    P2160 = 1.0 / 2.160
    au = dict(metal=MaterialId.GOLD)  # Au/Pd: gold stands in for the alloy
    old, new = dict(edge_nm=10.0, edge_corr_nm=60.0, wavy_nm=40.0, wavy_um=1.2), dict(
        edge_nm=3.0, edge_corr_nm=80.0, wavy_nm=6.0, wavy_um=3.0)
    # The current gratings (607, 603, 677, 673; Ted Pella's images): square wells between
    # narrow raised lines, steep walls, shadowed along the diagonal at a low angle, so each
    # well floor has a bare L-shaped shadow ~1/4 of it wide and the rest a grainy coat.
    # The older style (607-A, 603-A): a softer, rougher, wavier embossed relief.
    # Relief, line profile and coat fitted to Ted Pella's images (unit-cell averages and their
    # harmonics): the lines are sawtooth (a steep wall and a gentle slope, as a ruled master).
    if number in ("607", "603"):
        g = WaffleGrating(P2160, depth_nm=90.0, line_fraction=0.24, ramp_nm=8.0, back_ramp_nm=75.0, **new)
        sph = LatexSpheres(262.0, 0.15) if number == "603" else None
        return ShadowedReplicaStructure(g, rough_nm=2.5, rough_um=0.1, crumple_nm=6.0, metal_nm=5.0,
                                        elevation_deg=20.0, spheres=sph, spheres_shadowed=False, **au)
    if number in ("607-A", "603-A"):
        g = WaffleGrating(P2160, depth_nm=25.0, line_fraction=0.18, ramp_nm=45.0, **old)
        sph = LatexSpheres(262.0, 0.15) if number == "603-A" else None
        return ShadowedReplicaStructure(g, rough_nm=7.0, rough_um=0.1, elevation_deg=20.0, spheres=sph,
                                        spheres_shadowed=False, **au)
    if number == "606":
        return ShadowedReplicaStructure(WaffleGrating(P2160, depth_nm=90.0, line_fraction=0.24, ramp_nm=8.0,
                                                      back_ramp_nm=75.0, both=False, **new),
                                        rough_nm=2.5, rough_um=0.1, crumple_nm=6.0, metal_nm=5.0,
                                        elevation_deg=20.0, **au)
    if number in ("677", "673"):
        g = WaffleGrating(0.5, depth_nm=60.0, line_fraction=0.12, ramp_nm=70.0, back_ramp_nm=10.0, **new)
        sph = LatexSpheres(262.0, 0.15) if number == "673" else None
        return ShadowedReplicaStructure(g, rough_nm=2.0, rough_um=0.1, crumple_nm=6.0, metal_nm=12.0,
                                        elevation_deg=20.0, spheres=sph, spheres_shadowed=False, **au)
    if number == "628-B":
        # a heavy, low-angle gold shadow: shadows ~3.5 diameters long
        return ShadowedReplicaStructure(None, base_nm=15.0, rough_nm=1.5, crumple_nm=4.0, metal_nm=15.0,
                                        elevation_deg=16.0, spheres=LatexSpheres(204.0, 0.8, 0.02),
                                        island_nm=10.0, coverage=0.7, **au)
    p = PRODUCTS[number]
    if p.family == "Latex spheres":
        d = p.numbers["sphere_nm"]
        density = min(20.0, 0.05 / (3.1416 * (d / 2000.0) ** 2))  # ~5% of the film covered
        return ShadowedReplicaStructure(None, base_nm=15.0, rough_nm=0.0, crumple_nm=0.0, metal_nm=0.0,
                                        spheres=LatexSpheres(d, density, p.model["cv"], cluster=4.0))
    if number == "619":
        return PolycrystalFilmStructure(MaterialId.ALUMINUM, 31.0, 15.0)
    if number == "625":
        return MoO3LathsStructure()
    if number == "646":
        return OrientedFoilStructure(MaterialId.GOLD, 15.0)
    if number == "645":
        return HoleyCarbonStructure(HoleyFilm(12.0), extra=CarbonBlackParticles())
    if number == "675":
        return MagicalStructure()
    if number == "612":
        return CatalaseStructure()
    if number == "608":
        return FerritinStructure()
    if number == "613":
        return HoleyCarbonStructure(HoleyFilm(15.0), MaterialId.GOLD, deposit_nm=4.0, island_nm=7.0,
                                    coverage=0.76, grain_nm=5.0)
    if number == "611":
        return HoleyCarbonStructure(HoleyFilm(12.0), MaterialId.PLATINUM, deposit_nm=0.6, island_nm=1.4,
                                    coverage=0.24, grain_nm=1.4)
    if number == "609":
        return HoleyCarbonStructure(HoleyFilm(25.0))
    if number == "638":
        return HoleyCarbonStructure(HoleyFilm(15.0), MaterialId.GOLD, deposit_nm=1.5, island_nm=8.0, coverage=0.55,
                                    grain_nm=6.0, extra=CarbonBlackParticles(cluster_cell_um=1.5, cluster_fraction=0.5,
                                                                             per_cluster=120.0))
    raise KeyError(f"no structure for product {number!r}")


def mean_thickness_nm(number: str) -> float:
    """Area-mean carbon-equivalent thickness of product *number* (its low-magnification look)."""
    return build_structure(number).mean_nm()


def preset_name(number: str) -> str:
    return f"Ted Pella {number} - {PRODUCTS[number].name}"
