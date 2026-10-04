# EBSD in an SEM

The twin can be an SEM with a Direct Electron camera as its electron backscatter diffraction
(EBSD) detector: a bulk polished specimen tilted to 70°, scanned point by point like 4D-STEM,
one Kikuchi pattern per scan point. de-twin produces the patterns and the ground truth (phase,
grain, orientation, pattern centre per point); index them with the tool of your choice.

```python
from de_twin import DigitalTwin
from de_twin.state import ScanRequest

twin = DigitalTwin("EBSD duplex steel", camera="Celeritas")
twin.configure_ebsd(beam_kv=20.0, sample_tilt_deg=70.0, pcx=0.5, pcy=0.3, pcz=0.6,
                    probe_current_na=0.1)
req = twin.request(hw_binning=(4, 4), frame_time_s=1e-3, total_frames=100 * 80,
                   scan=ScanRequest(enabled=True, size=(100, 80), step_um=0.5, dwell_s=1e-3))

for raw, meta in twin.frames(req):        # raw camera frames, 256 x 256, as DE-Server sends them
    ...
patterns = twin.ebsd_patterns(req)         # noiseless stack (80, 100, 256, 256), e- / binned px / s
xmap = twin.ground_truth(req)              # orix CrystalMap: phase, orientation, grain_id, pcx/pcy/pcz,
                                           # band_contrast per scan point (x, y in um)
```

`configure_ebsd` sets `OpticsConfig.ebsd`, an {class}`~de_twin.ebsd.EbsdGeometry`: a frozen
dataclass of plain numbers (`to_dict` / `from_dict` / `replace`, `describe()` for ranges), so a
GUI can persist it. With it set, every acquisition is `RenderMode.EBSD`; `enabled=False`
returns the twin to a TEM. The beam energy is the column's high tension (`beam_kv`; 5–30 kV).
The deapi face exposes the geometry as `Simulator EBSD ...` properties.

The pattern stack is `ny * nx * h * w * 4` bytes; pass `out=np.memmap(...)` (or `dtype`) to
`ebsd_patterns` for large scans.

## Specimens

`de_twin.specimen.BULK_PRESETS`, by name:

| preset | phases | microstructure |
| --- | --- | --- |
| `EBSD Ni polycrystal` | FCC Ni | 8 µm equiaxed grains, random texture |
| `EBSD Al cube texture` | FCC Al | 20 µm grains, {001}<100> cube texture (10°), polishing scratches |
| `EBSD ferrite gamma fibre` | BCC Fe | 10 µm grains, <111>‖ND fibre (10°) |
| `EBSD duplex steel` | BCC ferrite 55 %, FCC austenite 45 % | 6 µm grains, scratches, light damage |
| `EBSD Si (001) wafer` | Si | single crystal, [001] normal, [110] along x: pattern-centre calibration |

A {class}`~de_twin.specimen.BulkSurface` is an infinite polished surface: a Voronoi
tessellation of a hashed jittered lattice (deterministic, evaluated point by point), a phase
per grain from the phase fractions, and an orientation per grain drawn from the phase's
{class}`~de_twin.crystal.Texture`, unique to the grain. Build your own with
{class}`~de_twin.specimen.BulkSurfaceConfig`. A TEM specimen also works in EBSD mode (its
field map's phases and grain orientations, no boundary mixing).

## Conventions

- **Sample frame** (orientations are reported in it): `x_s` = scan x (right), `y_s` = scan y
  (down), `z_s` into the sample. The tilt axis is `x_s`; the detector is at the top of the SEM
  image. Orientations are orix rotations: crystal→sample matrix `G = R.to_matrix().T`, exactly
  as everywhere else in de-twin (`crystal_to_lab`).
- **Detector frame**: `x_d` along pattern columns, `y_d` down the rows, `z_d` from the source to
  the screen. The pattern is the screen as seen from the sample: the bright forward-scatter
  region is at the bottom.
- **Pattern centre: Bruker convention.** `pcx` from the left in units of the sensor width,
  `pcy` from the top in units of the sensor height, `pcz` = detector distance / sensor height,
  all for the whole unbinned sensor (binning and ROI leave them unchanged).
  `de_twin.ebsd.convert_pc` gives the EDAX/TSL, Oxford and EMsoft (v5) forms.
- **The pattern centre follows the beam**: `EbsdGeometry.pcx/pcy/pcz` are for the beam on the
  optic axis (scan centre) at `calibration_wd_mm`. A beam offset `(dx, dy)` on the sample moves
  the source by `S2D @ (dx, dy, 0)`: `pcx += dx/(W p)`, `pcy -= sin(σ−θ) dy/(H p)`,
  `pcz += cos(σ−θ) dy/(H p)` (σ sample tilt, θ detector tilt, p pixel pitch). A working
  distance change moves it too. The ground truth carries the pattern centre of every point.
- **Scan step** is on the tilt-corrected sample surface (`ScanRequest.step_um`, else
  `EbsdGeometry.step_um`).

## Model

- **Master patterns**: kinematic, one per phase and beam energy, built from the crystal
  library's structure factors (Lobato scattering factors, Debye–Waller) and cached
  process-wide. Every ±g pair gives a band of width 2θ_B and weight |F_g|²; higher orders nest
  inside. Band edges are blurred by the backscatter energy spread (`energy_spread_frac`) and the
  screen's angular blur, with an excess line inside and a deficiency outside each edge.
  Standardised, the bright tail log-compressed, stored as uint8 on a square-Lambert hemisphere
  (kinematic patterns are centrosymmetric): 0.9 MB per phase at 0.15°, 0.3–0.6 s to build.
- **Projection**: per scan point, one numba pass over the binned detector grid: the gnomonic
  direction of each pixel, rotated into the crystal by the grain orientation, the sample tilt and
  the detector tilt, with the point's own pattern centre.
- **Signal**: backscatter yield (Arnal, by Z and tilt) × probe current × an angular
  distribution (a cosine about the surface normal plus a forward lobe about the specular
  direction, sharper at higher kV) × each pixel's solid angle (cos³γ / DD²), shadowed below the
  sample surface; bands modulate it by `1 + band_contrast × quality × m`.
- **Grain boundaries**: within the interaction footprint (0.02 × the Kanaya–Okayama range across
  the tilt axis, 1/cos σ longer along it; ~30 × 80 nm for Ni at 20 kV) the neighbouring grain's
  pattern is mixed in by the fraction of the footprint across the boundary. Surface damage and
  polishing scratches lower `quality`.
- **Noise**: the camera's own detector model. At 20 kV an electron stops in the sensor and
  deposits all its energy (`CameraModel.full_stop_kv`), ~1600 ADU on a Celeritas: an
  integrating camera holds about one electron per pixel per frame, so EBSD runs at low current
  (default 0.1 nA) or binned.

What it does not have, compared with EMsoft dynamical master patterns: no Bloch-wave
intensities or channelling (band intensities are kinematic |F|², not dynamical), no Monte-Carlo
depth and energy distribution of the backscattered electrons (one effective energy and a fixed
edge blur), no HOLZ rings or fine zone-axis structure, no anomalous absorption, and excess /
deficiency edges that are symmetric and fixed in the crystal rather than set by the scattering
geometry. The background is a smooth heuristic, not a Monte-Carlo result.

## Speed

On a 48-thread workstation: 0.8–1.7 ms per noiseless pattern at 256² or 512² (binned), 1 ms per
noisy 256² frame and 3 ms per 512² frame through the detector model (~1000 and ~310 frames/s).
Master patterns are built on first use of a phase and energy.

The test suite checks the geometry (zone axes and band edges where the geometry puts them, the
pattern centre moving with the beam), round-trips the ground truth, and indexes noisy frames
with a test-only dictionary + refinement to well under 1° (~10 e⁻ per binned pixel).
