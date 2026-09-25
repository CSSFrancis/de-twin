"""Render-side options (the renderer knobs of VirtualSpecimenProps plus the twin's
wave-optical TEM model)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderConfig:
    # ---- TEM imaging ------------------------------------------------------
    tem_model: str = "physical"  # "physical" (exit wave + CTF) | "legacy" (C++ port)
    diffraction_contrast_scale: float = 1.0  # scales the Bragg loss from the bright-field beam
    mip_phase: bool = True  # mean-inner-potential phase (gives Fresnel fringes)
    edge_taper_nm: float = 1.0  # Gaussian roughness of specimen edges applied to the MIP phase
    refraction_loss: bool = True  # electrons refracted beyond the imaging band by steep edges are lost
    phase_texture_scale: float = 1.0  # amorphous atomic-granularity phase (gives Thon rings)
    texture_bandlimit_nm: float = 0.07  # Gaussian sigma: atomic form-factor band limit
    amplitude_contrast: float = 0.07  # absorptive part of the amorphous texture (w)
    lattice_fringes: bool = True
    lattice_phase_rad: float = 0.15  # total phase amplitude of resolved lattice fringes
    #: TEM imaging renders a view this fraction of the field larger on each side and
    #: serves a stage move that stays inside the margin by cropping it: the image moves
    #: rigidly with the specimen, so a joystick nudge costs a crop, not a render. 0 off.
    pan_margin: float = 0.15
    #: Interactive use (dragging the stage, zooming): while the view keeps changing, a
    #: view the cache cannot crop is rendered at up to `preview_side`² and upsampled
    #: (~35-70 ms rather than ~0.3-0.9 s); the first time the same view is asked for
    #: again — the next frame after you stop — it is rendered in full. Off by default,
    #: so a library caller always gets the full render.
    interactive: bool = False
    preview_side: int = 256
    #: How long (wall-clock seconds) the view must stay put before it counts as stopped
    #: and is rendered in full. Live view asks for several frames per drag step, so
    #: "asked for the same view twice" is not "stopped".
    settle_s: float = 0.3
    illumination_profile: bool = True  # draw the edge of the illuminated disk when it is in view
    texture_seed_salt: int = 0x7E57

    # ---- diffraction (de_twin.crystal library + DiffractionCache) -------
    diffraction_cache_mb: int = 256
    max_g_inv_nm: float = 25.0  # reciprocal-lattice extent (raise it for HOLZ reflections)
    film_halo: bool = True  # support-film halo under every crystalline pattern
    scaled_diffraction_saturation: bool = True  # per-material amorphous saturation thickness
    diffuse_scattering: bool = True  # (1-T) redistributed as a screened-Rutherford background
    beam_stop_radius_px: float = 0.0  # SAED beam stop, detector pixels; 0 = none

    # ---- STEM -----------------------------------------------------------
    probe_footprint_blend: bool = True
    add_descan: bool = False
    descan_ramp_scale: float = 1.0
    descan_ramp_px: tuple[float, float, float] = (6.0, 4.0, 4.0)  # C++ legacy x span, y span, diag
    intensity_jitter: bool = True  # C++ per-scan-point 0.9..1.1 fluctuation
    park_on_feature: bool = False  # C++ default was on; off keeps the probe on the optic axis

    # ---- coherent 4D-STEM (render/coherent.py) ----------------------------
    # "coherent": probe wavefunction x transmission function, |FFT|^2 (ptychography / tcBF work);
    # "kinematic": cached disk patterns (fast, no interference);
    # "auto": coherent when the simulation grid fits coherent_max_grid without cutting the probe
    #   window and the binned pattern is <= coherent_auto_max_pattern_px, or the scan step is
    #   below 2x the probe size (ptychographic overlap); else kinematic.
    stem_model: str = "auto"
    coherent_max_grid: int = 1024  # largest simulation grid side per pattern
    coherent_auto_max_pattern_px: int = 256 * 256
    coherent_probe_tail: float = 6.0  # window margin beyond the geometric probe radius, in lambda/alpha
    coherent_mode_power: float = 0.98  # partial coherence: keep probe modes up to this power fraction
    coherent_max_modes: int = 4
    coherent_focal_samples: int = 0  # 0 = auto (1/3/5 Gauss-Hermite samples of the focal spread)
    coherent_focal_spread: bool = True  # temporal partial coherence (OpticsState.focal_spread_nm)
    coherent_source_size: bool = True  # spatial partial coherence (OpticsState.source_size_nm)
    coherent_beam_tilt: bool = True  # beam tilt enters chi (coma etc.) and shifts the probe
    coherent_incoherent_scattering: bool = True  # Bragg disks + diffuse background added incoherently
    # t(r) band limit K_t, 1/nm: scattering to |g| <= K_t is coherent, beyond it incoherent. 0 = auto:
    # k_det + alpha/lambda (everything that can reach the detector is coherent)
    coherent_object_bandwidth_inv_nm: float = 0.0
    coherent_field_max_px: int = 36_000_000  # larger scan fields build transmission tiles per block
    coherent_cache_mb: int = 512  # computed pattern blocks kept for frame-by-frame rendering

    # ---- caching --------------------------------------------------------
    fieldmap_cache_size: int = 4
    time_quantum_s: float = 0.1  # re-rasterise a time-dependent specimen at most this often
