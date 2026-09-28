"""Fused numba kernels for the TEM imaging path.

Each kernel does in one pass over the raster what the NumPy reference in
:mod:`de_twin.render.tem` does in several: the Bragg pixel keys and loss gather, the
mean-inner-potential phase, the exit wave (amplitude, absorption, texture, refraction loss,
phase), the transfer function of the usual aberration set, and the spectrum product and
``|psi|^2``. Same arithmetic in float32; they differ from the reference by float rounding
(tested). ``AVAILABLE`` is False without numba, and the reference is used.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import numba as nb

    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False


def _njit(**kw):
    if not AVAILABLE:
        return lambda f: f
    return nb.njit(cache=True, nogil=True, **kw)


_prange = nb.prange if AVAILABLE else range


# ------------------------------------------------------------------ Bragg pixels
@_njit()
def _tbin(t):
    """``thickness_bin_for``: rint(t / 2 nm) (half to even), clipped to 0..250."""
    x = t / 2.0
    r = math.floor(x + 0.5)
    if r - x == 0.5 and r % 2.0 != 0.0:
        r -= 1.0
    if r < 0.0:
        r = 0.0
    if r > 250.0:
        r = 250.0
    return int(r)


@_njit(parallel=True)
def bragg_pixels(grain_id, material_id, thickness, f_t, n, nb_, gpm, gid, pix, maxbin):
    """Per pixel: own grain (-1 none) and the key ``grain * nb_ + thickness bin`` (-1);
    ``maxbin[c, g]``: the largest thickness bin of grain g in row chunk c."""
    ny, nx = grain_id.shape
    nc = maxbin.shape[0]
    for c in _prange(nc):
        r0 = ny * c // nc
        r1 = ny * (c + 1) // nc
        for i in range(r0, r1):
            for j in range(nx):
                g = np.int64(grain_id[i, j])
                if g >= 0 and g < n and g // gpm == material_id[i, j]:
                    tb = _tbin(np.float64(np.float32(thickness[i, j]) * f_t))
                    gid[i, j] = g
                    pix[i, j] = g * nb_ + tb
                    if tb > maxbin[c, g]:
                        maxbin[c, g] = tb
                else:
                    gid[i, j] = -1
                    pix[i, j] = -1


@_njit(parallel=True)
def gather(lut, pix, out):
    ny, nx = pix.shape
    for i in _prange(ny):
        for j in range(nx):
            p = pix[i, j]
            out[i, j] = lut[p] if p >= 0 else np.float32(0.0)


# ------------------------------------------------------------------ exit wave
@_njit(parallel=True)
def mip_phase(mat, thick, umat, uthick, has_under, f_t, sigma, mip_v, out):
    ny, nx = mat.shape
    for i in _prange(ny):
        for j in range(nx):
            v = sigma * mip_v[mat[i, j]] * (np.float32(thick[i, j]) * f_t)
            if has_under:
                v = v + sigma * mip_v[umat[i, j]] * (uthick[i, j] * f_t)
            out[i, j] = v


@_njit(parallel=True)
def refraction_qmax(mip, inv_gc2):
    ny, nx = mip.shape
    rowmax = np.zeros(ny, np.float32)
    for i in _prange(ny):
        m = np.float32(0.0)
        for j in range(nx):
            gy = np.float32(0.5) * (mip[i + 1, j] - mip[i - 1, j]) if 0 < i < ny - 1 else np.float32(0.0)
            gx = np.float32(0.5) * (mip[i, j + 1] - mip[i, j - 1]) if 0 < j < nx - 1 else np.float32(0.0)
            q = (gx * gx + gy * gy) * inv_gc2
            if q > m:
                m = q
        rowmax[i] = m
    return rowmax.max()


@_njit(parallel=True)
def exit_wave(mat, thick, umat, uthick, has_under, f_t, lam_abs, bragg, mip, has_mip, refraction,
              inv_gc2, noise, has_noise, amorphous, tex_k, mip_v, inv_dens, kappa, psi, t_out):
    ny, nx = mat.shape
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for i in _prange(ny):
        for j in range(nx):
            m = mat[i, j]
            t = np.float32(thick[i, j]) * f_t
            t_out[i, j] = t
            e = t / lam_abs[m]
            if has_under:
                e = e + uthick[i, j] * f_t / lam_abs[umat[i, j]]
            T = np.float32(math.exp(-e))
            I = T * (one - bragg[i, j])
            if I < zero:
                I = zero
            elif I > one:
                I = one
            phi = mip[i, j] if has_mip else zero
            amp = np.float32(math.sqrt(I))
            if refraction:
                gy = np.float32(0.5) * (mip[i + 1, j] - mip[i - 1, j]) if 0 < i < ny - 1 else zero
                gx = np.float32(0.5) * (mip[i, j + 1] - mip[i, j - 1]) if 0 < j < nx - 1 else zero
                q = (gx * gx + gy * gy) * inv_gc2
                amp = amp * np.float32(math.exp(-q * q))
            if has_noise:
                var = zero
                if amorphous[m]:
                    a = tex_k * mip_v[m]
                    var = a * a * t * inv_dens[m]
                if has_under:
                    um = umat[i, j]
                    a = tex_k * mip_v[um]
                    var = var + a * a * (uthick[i, j] * f_t) * inv_dens[um]
                tex = np.float32(math.sqrt(var)) * noise[i, j]
                phi = phi + tex
                if kappa != zero:
                    amp = amp * np.float32(math.exp(-kappa * tex))
            psi[i, j] = complex(amp * math.cos(phi), amp * math.sin(phi))


# ------------------------------------------------------------------ transfer
@_njit(parallel=True)
def transfer_c1a1c3c5(kx1, ky1, tx, ty, radial, poly3, poly5, a_re, a_im, g1, g3, g5, ga_re, ga_im,
                      c0, gx0, gy0, es, et, kt2, kap2, H):
    """H(k) of an aberration set of C1, A1, C3 and C5 (`tem.transfer_function`)."""
    ny = ky1.shape[0]
    nx = kx1.shape[0]
    for i in _prange(ny):
        KY = np.float32(ky1[i] + ty)
        for j in range(nx):
            KX = np.float32(kx1[j] + tx)
            k2 = KX * KX + KY * KY
            chi = k2 * (radial + k2 * (poly3 + poly5 * k2)) + a_re * (KX * KX - KY * KY) + a_im * (KX * KY)
            rad = g1 + k2 * (g3 + g5 * k2)
            gx = rad * KX + ga_re * KX + ga_im * KY - gx0
            gy = rad * KY - ga_re * KY + ga_im * KX - gy0
            chi = chi - c0
            k2t = k2 - kt2
            if kap2 > 0.0 and k2 > kap2:
                H[i, j] = 0.0
                continue
            env = math.exp(-(es * (gx * gx + gy * gy) + et * k2t * k2t))
            H[i, j] = complex(env * math.cos(chi), -env * math.sin(chi))


@_njit(parallel=True)
def spectrum_product(spec, H, has_h, ry, rx, has_ramp, out):
    ny, nx = spec.shape
    for i in _prange(ny):
        for j in range(nx):
            v = spec[i, j]
            if has_h:
                v = v * H[i, j]
            if has_ramp:
                v = v * ry[i] * rx[j]
            out[i, j] = v


@_njit(parallel=True)
def intensity(psi, out):
    ny, nx = psi.shape
    for i in _prange(ny):
        for j in range(nx):
            v = psi[i, j]
            out[i, j] = v.real * v.real + v.imag * v.imag
