"""SerialEM calibrations to and from a column definition.

A twin's column (`OpticsConfig.realism`) can be written out as the calibrations SerialEM would
measure on it, and a column can be defined from a real microscope's SerialEM calibrations, so
the twin behaves like that microscope. Fitting or refining a definition from acquired data is
not done here (that is Ground Crew's / de_autopilot's job); this module only converts.

Files (formats from SerialEM's ``ParameterIO.cpp``):

``SerialEMcalibrations.txt``
    ``ImageShiftMatrix`` (count, then ``magInd camera xpx xpy ypx ypy mag``): camera pixels
    the image moves per image-shift unit; ``StageToCameraMatrix magInd camera xpx xpy ypx ypy
    focus mag``: camera pixels per stage micrometre; ``ImageShiftOffsets`` (count, then
    ``magInd 0 isx isy``): the image shift that re-centres each magnification;
    ``BeamShiftCalibration magInd xpx xpy ypx ypy alpha probe retain mag``: beam shift per
    image shift; ``CrossoverIntensity spot micro nano``; ``HighFocusMagCal spot probe
    defocus intensity scale rotation crossover aperture magInd``; ``FocusCalibration magInd
    camera slopeX slopeY beamTilt nPoints direction probe alpha mag`` then ``defocus dx dy``
    lines.
``SerialEMproperties.txt`` (a camera's block)
    ``RotationAndPixel magInd deltaRotation rotation pixel_nm``.

Conventions. Camera coordinates are (x right, y down) raster pixels of the unbinned camera; a
matrix ``[[xpx, xpy], [ypx, ypy]]`` maps (x, y) to (x', y') = (xpx x + xpy y, ypx x + ypy y).
SerialEM's image rotation is the angle of ``SpecimenToCamera = R(rotation) / pixel``. Where
SerialEM's own camera frame differs in handedness from this one, the rotations change sign
together, so a round trip is exact and relations between magnifications are preserved; check
the sign once against a real microscope before trusting absolute angles.

Magnification indices: SerialEM numbers the scope's magnifications from 1. The twin's own are
its imaging ladder (LowMAG then MAG1) numbered from 1 (`twin_mag_table`); a real scope's table
can be passed as ``{index: magnification}``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from .realism import ColumnRealism

#: The beam tilt (mrad) and defocus points of the focus calibration written out.
FOCUS_CAL_TILT_MRAD = 3.0
FOCUS_CAL_DEFOCUS_UM = (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
#: The defocus values of the high-focus magnification calibration written out.
HIGH_FOCUS_DEFOCUS_UM = (-50.0, -100.0, -200.0, -300.0)


def twin_mag_table() -> dict[int, float]:
    """The twin column's imaging magnifications, numbered from 1 as SerialEM numbers a scope's."""
    from ..column import ladders as L

    return {i + 1: float(m) for i, m in enumerate(L.LOWMAG_MAGS + L.MAG1_MAGS)}


def _mode_for(mag: float) -> str:
    from ..column import ladders as L

    return "LowMAG" if mag <= L.LOWMAG_MAGS[-1] else "MAG1"


def _rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


# ------------------------------------------------------------------ the column definition
class ColumnDefinition:
    """A column defined by tables (from calibrations) rather than drawn from a seed: the same
    interface as :class:`~de_twin.optics.realism.ColumnRealism`, so it goes in
    ``OpticsConfig(realism=...)``.

    Per magnification: the image rotation (rad), the true pixel size (nm, for the camera it was
    calibrated on) against the twin's nominal one, the image-shift matrix (specimen um per
    unit) and the magnification's image offset (um). One beam-shift matrix, crossovers per
    (spot, probe), and the high-defocus scale / rotation per micrometre. What the tables do not
    give is ideal. Magnifications between entries take the nearest (in log magnification).
    Compared and hashed by identity (it holds mutable tables).
    """

    def __init__(self):
        self.rotation: dict = {}  # mag -> rad
        self.pixel_nm: dict = {}  # mag -> true pixel (nm)
        self.nominal_pixel_nm: dict = {}  # mag -> the twin's nominal (nm)
        self.is_matrices: dict = {}  # mag -> 2x2
        self.mag_offsets: dict = {}  # mag -> (x, y) um
        self.bs: Optional[np.ndarray] = None
        self.crossovers: dict = {}  # (spot, probe) -> intensity
        self.crossover_intensity = ColumnRealism.crossover_intensity
        self.backlash_um = 0.0
        self.is_coma_mrad_per_um = 0.0
        self.is_astig_nm_per_um = 0.0
        self.hd_scale_per_um = 0.0
        self.hd_rotation_deg_per_um = 0.0

    def defocus_scale(self, defocus_um: float) -> float:
        return 1.0 + self.hd_scale_per_um * abs(float(defocus_um))

    def defocus_rotation_rad(self, defocus_um: float) -> float:
        return math.radians(self.hd_rotation_deg_per_um * float(defocus_um))

    def truth(self, mag_mode, mag: float) -> dict:
        return {
            "image_rotation_deg": math.degrees(self.rotation_rad(mag_mode, mag)),
            "pixel_scale": self.pixel_scale(mag_mode, mag),
            "is_matrix_um_per_unit": self.is_matrix(mag_mode, mag).tolist(),
            "mag_offset_um": self.mag_offset_um(mag_mode, mag),
            "backlash_um": self.backlash_um,
            "bs_matrix_um_per_unit": self.bs_matrix().tolist(),
        }

    def _near(self, table: dict, mag: float):
        if not table:
            return None
        k = min(table, key=lambda m: abs(math.log(max(m, 1e-9)) - math.log(max(float(mag), 1e-9))))
        return table[k]

    def rotation_rad(self, mag_mode, mag: float) -> float:
        v = self._near(self.rotation, mag)
        return 0.0 if v is None else float(v)

    def pixel_scale(self, mag_mode, mag: float) -> float:
        p = self._near(self.pixel_nm, mag)
        n = self._near(self.nominal_pixel_nm, mag)
        return 1.0 if not p or not n else float(p) / float(n)

    def is_matrix(self, mag_mode, mag: float) -> np.ndarray:
        v = self._near(self.is_matrices, mag)
        return np.eye(2) if v is None else np.asarray(v, float)

    def mag_offset_um(self, mag_mode, mag: float) -> tuple[float, float]:
        v = self._near(self.mag_offsets, mag)
        return (0.0, 0.0) if v is None else (float(v[0]), float(v[1]))

    def bs_matrix(self) -> np.ndarray:
        return np.eye(2) if self.bs is None else np.asarray(self.bs, float)

    def crossover(self, spot: int, probe_mode: int = 0) -> float:
        if (int(spot), int(probe_mode)) in self.crossovers:
            return float(self.crossovers[(int(spot), int(probe_mode))])
        if self.crossovers:
            return float(np.mean(list(self.crossovers.values())))
        return float(self.crossover_intensity)

    def is_tilt_mrad(self, isx_um: float, isy_um: float) -> tuple[float, float]:
        k = self.is_coma_mrad_per_um
        return k * isx_um, k * isy_um

    def is_astig_nm(self, isx_um: float, isy_um: float) -> complex:
        return self.is_astig_nm_per_um * complex(isx_um, isy_um)


# ------------------------------------------------------------------ twin -> SerialEM
def _nominal_pixel_nm(mag: float, camera, calibration=None) -> float:
    from ..state import MicroscopeState
    from .calibration import Calibration

    cal = calibration or Calibration.default()
    s = MicroscopeState(magnification=float(mag), mag_mode=_mode_for(mag))
    return float(cal.specimen_pixel_nm(s, camera))


def _camera_mats(realism: ColumnRealism, mag: float, nominal_nm: float):
    """(StageToCamera, IStoCamera, rotation rad, true pixel nm) of one magnification."""
    mode = _mode_for(mag)
    theta = realism.rotation_rad(mode, mag)
    p_um = nominal_nm * realism.pixel_scale(mode, mag) / 1000.0
    to_cam = _rot(-theta) / p_um  # specimen um -> camera px (image motion of a feature)
    stage = to_cam  # a stage step moves the image with the specimen
    is_ = -to_cam @ realism.is_matrix(mode, mag)  # image shift moves the view, the image the other way
    return stage, is_, theta, p_um * 1000.0


def to_serialem(realism: ColumnRealism, camera, calibrations_path, properties_path=None, *,
                mag_table: Optional[dict] = None, calibration=None, spots=(1, 2, 3, 4, 5),
                reference_mag: float = 20000.0, camera_index: int = 0) -> None:
    """Write the calibrations SerialEM would measure on a column (and, with
    *properties_path*, the camera's ``RotationAndPixel`` lines)."""
    from ..detector import camera as camera_by_name

    cam = camera_by_name(camera) if isinstance(camera, str) else camera
    mags = mag_table or twin_mag_table()
    lines = ["SerialEMCalibrations"]
    rows, stage_rows, offs, rot_rows = [], [], [], []
    for ind, mag in sorted(mags.items()):
        nominal = _nominal_pixel_nm(mag, cam, calibration)
        stage, is_, theta, p_nm = _camera_mats(realism, mag, nominal)
        rows.append(f"{ind} {camera_index} {is_[0, 0]:.9g} {is_[0, 1]:.9g} {is_[1, 0]:.9g} {is_[1, 1]:.9g}   {int(round(mag))}")
        stage_rows.append(f"StageToCameraMatrix {ind} {camera_index} {stage[0, 0]:.9g} {stage[0, 1]:.9g} "
                          f"{stage[1, 0]:.9g} {stage[1, 1]:.9g}   0.000000   {int(round(mag))}")
        mode = _mode_for(mag)
        o = np.asarray(realism.mag_offset_um(mode, mag))
        if np.any(o):
            isx, isy = -np.linalg.solve(realism.is_matrix(mode, mag), o)
            offs.append(f"{ind} 0 {isx:.9g} {isy:.9g}")
        rot_rows.append(f"RotationAndPixel {ind} 0 {math.degrees(-theta):.9g} {p_nm:.9g}")
    lines.append(f"ImageShiftMatrix {len(rows)}")
    lines += rows
    lines += stage_rows
    if offs:
        lines.append(f"ImageShiftOffsets {len(offs)}")
        lines += offs
    # beam shift that keeps the beam on the imaged area per image-shift unit: B^-1 M
    B = realism.bs_matrix()
    for ind, mag in sorted(mags.items()):
        m = np.linalg.solve(B, realism.is_matrix(_mode_for(mag), mag))
        lines.append(f"BeamShiftCalibration {ind} {m[0, 0]:.9g} {m[0, 1]:.9g} {m[1, 0]:.9g} {m[1, 1]:.9g} "
                     f"-999 1 0  {int(round(mag))}")
    for spot in spots:
        lines.append(f"CrossoverIntensity {spot} {realism.crossover(spot, 1):.9g} {realism.crossover(spot, 0):.9g}")
    # high-focus magnification: image scale and rotation relative to focus
    ref = min(mags, key=lambda i: abs(mags[i] - reference_mag))
    for df in HIGH_FOCUS_DEFOCUS_UM:
        scale = 1.0 / realism.defocus_scale(df)
        rot = -math.degrees(realism.defocus_rotation_rad(df))
        lines.append(f"HighFocusMagCal 3 1 {df:.6f} 0.500000 {scale:.9g} {rot:.9g} "
                     f"{realism.crossover(3, 1):.9g} 0 {ref}")
    # autofocus: image displacement (camera px) under +/- beam tilt, per defocus
    mag = mags[ref]
    stage, is_, theta, p_nm = _camera_mats(realism, mag, _nominal_pixel_nm(mag, cam, calibration))
    d = _rot(-theta) @ np.array([1.0, 0.0])
    tau = FOCUS_CAL_TILT_MRAD * 1e-3
    pts = [(df, *(2.0 * df * 1000.0 * tau / p_nm * d)) for df in FOCUS_CAL_DEFOCUS_UM]
    slope = 2.0 * 1000.0 * tau / p_nm * d
    lines.append(f"FocusCalibration {ref} {camera_index} {slope[0]:.9g} {slope[1]:.9g} "
                 f"{FOCUS_CAL_TILT_MRAD:.2f} {len(pts)} 0 1 -999   {int(round(mag))}")
    lines += [f"{a:.6f} {b:.9g} {c:.9g}" for a, b, c in pts]
    Path(calibrations_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if properties_path is not None:
        Path(properties_path).write_text(
            "# RotationAndPixel lines for the camera's block in SerialEMproperties.txt\n"
            + "\n".join(rot_rows) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ SerialEM -> column
def _floats(parts):
    return [float(v) for v in parts]


def from_serialem(calibrations_path, properties_path=None, *, camera, mag_table: Optional[dict] = None,
                  calibration=None, camera_index: int = 0) -> ColumnDefinition:
    """A column definition from a microscope's SerialEM calibrations (and, if given, the
    camera's ``RotationAndPixel`` lines). Stage calibrations give each magnification's
    rotation and true pixel size where ``RotationAndPixel`` does not."""
    from ..detector import camera as camera_by_name

    cam = camera_by_name(camera) if isinstance(camera, str) else camera
    mags = mag_table or twin_mag_table()
    is_cam, stage_cam, offsets, bs_rows, cross, hf = {}, {}, {}, [], {}, []
    lines = Path(calibrations_path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        key = parts[0]
        if key == "ImageShiftMatrix":
            for _ in range(int(parts[1])):
                p = lines[i].split()
                i += 1
                if int(p[1]) == camera_index:
                    is_cam[int(p[0])] = np.array(_floats(p[2:6])).reshape(2, 2)
        elif key == "StageToCameraMatrix" and int(parts[2]) == camera_index:
            stage_cam[int(parts[1])] = np.array(_floats(parts[3:7])).reshape(2, 2)
        elif key == "ImageShiftOffsets":
            for _ in range(int(parts[1])):
                p = lines[i].split()
                i += 1
                if int(p[1]) == 0:
                    offsets[int(p[0])] = np.array(_floats(p[2:4]))
        elif key == "BeamShiftCalibration":
            bs_rows.append((int(parts[1]), np.array(_floats(parts[2:6])).reshape(2, 2)))
        elif key == "CrossoverIntensity":
            spot = int(parts[1])
            micro, nano = _floats(parts[2:4])
            if micro:
                cross[(spot, 1)] = micro
            if nano:
                cross[(spot, 0)] = nano
        elif key == "HighFocusMagCal":
            hf.append(_floats(parts[3:7]))  # defocus, intensity, scale, rotation
        elif key in ("FocusCalibration", "STEMfocusVersusZ"):
            i += int(parts[6] if key == "FocusCalibration" else parts[1])
    rot_pix = {}
    if properties_path is not None:
        for line in Path(properties_path).read_text(encoding="utf-8").splitlines():
            p = line.split()
            if p and p[0] == "RotationAndPixel":
                rot_pix[int(p[1])] = (math.radians(-float(p[3])), float(p[4]))

    d = ColumnDefinition()
    for ind, mag in mags.items():
        d.nominal_pixel_nm[mag] = _nominal_pixel_nm(mag, cam, calibration)
        if ind in stage_cam:
            s = stage_cam[ind]  # = R(-theta) / p_um
            p_um = 1.0 / math.sqrt(abs(np.linalg.det(s)))
            theta = -math.atan2(s[1, 0], s[0, 0])
            d.rotation[mag] = theta
            d.pixel_nm[mag] = p_um * 1000.0
        if ind in rot_pix:
            theta, p_nm = rot_pix[ind]
            d.rotation[mag] = theta
            if p_nm > 0:
                d.pixel_nm[mag] = p_nm
        if ind in is_cam and mag in d.rotation and mag in d.pixel_nm:
            to_cam = _rot(-d.rotation[mag]) / (d.pixel_nm[mag] / 1000.0)
            d.is_matrices[mag] = -np.linalg.solve(to_cam, is_cam[ind])
        if ind in offsets and mag in d.is_matrices:
            d.mag_offsets[mag] = tuple(-(d.is_matrices[mag] @ offsets[ind]))
    if bs_rows:
        # B = M (IS->BS)^-1, averaged over the magnifications that have both
        est = [d.is_matrices[mags[ind]] @ np.linalg.inv(m) for ind, m in bs_rows
               if ind in mags and mags[ind] in d.is_matrices]
        if est:
            d.bs = np.mean(est, axis=0)
    d.crossovers = cross
    if hf:
        hf = np.array(hf)
        dfs, scales, rots = hf[:, 0], hf[:, 2], hf[:, 3]
        # scale = 1 / (1 + k |df|), rotation = -r df
        k = np.linalg.lstsq(np.abs(dfs)[:, None], 1.0 / scales - 1.0, rcond=None)[0][0]
        r = np.linalg.lstsq(dfs[:, None], -rots, rcond=None)[0][0]
        d.hd_scale_per_um = float(k)
        d.hd_rotation_deg_per_um = float(r)
    else:
        d.hd_scale_per_um = 0.0
        d.hd_rotation_deg_per_um = 0.0
    return d
