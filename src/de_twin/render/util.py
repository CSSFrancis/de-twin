"""Small array helpers shared by the renderers (numpy; cupy-compatible in spirit)."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..hashing import hash_seed, mix_cell

TEXTURE_KIND = 48  # twin-only hash kind for world-locked textures
_TILE = 128


_POOL: ThreadPoolExecutor | None = None


def pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(16, (os.cpu_count() or 2))))
    return _POOL


def parallel_rows(fn, nrows: int, min_chunk: int = 64) -> None:
    """Run ``fn(r0, r1)`` over row chunks on the shared thread pool (numpy releases the GIL)."""
    workers = pool()._max_workers
    n = max(1, min(workers * 2, nrows // max(1, min_chunk)))
    if n <= 1:
        fn(0, nrows)
        return
    edges = np.linspace(0, nrows, n + 1).astype(int)
    list(pool().map(lambda i: fn(int(edges[i]), int(edges[i + 1])), range(n)))


def upsample2x_scaled(a: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Threaded pixel-centre bilinear 2x upsampling (edge clamped) times ``scale``."""
    a = np.asarray(a, np.float32)
    ny, nx = a.shape
    p = np.pad(a, 1, mode="edge")
    out = np.empty((2 * ny, 2 * nx), np.float32)
    c0 = np.float32(0.75 * scale)
    c1 = np.float32(0.25 * scale)

    def work(r0, r1):
        mid = p[r0 + 1:r1 + 1]
        up = np.empty((2 * (r1 - r0), nx + 2), np.float32)
        np.multiply(mid, c0, out=up[0::2])
        up[0::2] += c1 * p[r0:r1]
        np.multiply(mid, c0, out=up[1::2])
        up[1::2] += c1 * p[r0 + 2:r1 + 2]
        o = out[2 * r0:2 * r1]
        np.multiply(up[:, 1:-1], 0.75, out=o[:, 0::2])
        o[:, 0::2] += 0.25 * up[:, :-2]
        np.multiply(up[:, 1:-1], 0.75, out=o[:, 1::2])
        o[:, 1::2] += 0.25 * up[:, 2:]

    parallel_rows(work, ny, 32)
    return out


def upsample2x(a: np.ndarray) -> np.ndarray:
    """Pixel-centre-aligned bilinear 2x upsampling with edge clamping (separable, fast)."""
    a = np.asarray(a, np.float32)
    # rows
    up = np.empty((a.shape[0] * 2, a.shape[1]), np.float32)
    prev = np.concatenate([a[:1], a[:-1]], axis=0)
    nxt = np.concatenate([a[1:], a[-1:]], axis=0)
    up[0::2] = 0.75 * a + 0.25 * prev
    up[1::2] = 0.75 * a + 0.25 * nxt
    out = np.empty((up.shape[0], a.shape[1] * 2), np.float32)
    prev = np.concatenate([up[:, :1], up[:, :-1]], axis=1)
    nxt = np.concatenate([up[:, 1:], up[:, -1:]], axis=1)
    out[:, 0::2] = 0.75 * up + 0.25 * prev
    out[:, 1::2] = 0.75 * up + 0.25 * nxt
    return out


def upsample_to(a: np.ndarray, factor: int, shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Upsample by a power-of-two ``factor``, multiply by ``scale``, crop/edge-pad to ``shape``."""
    out = a
    f = int(factor)
    first = True
    while f > 1:
        out = upsample2x_scaled(out, scale if first else 1.0)
        first = False
        f //= 2
    if first:
        out = out * np.float32(scale)
    return fit_shape(out, shape)


def fit_shape(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if a.shape == (h, w):
        return a
    a = a[:h, :w]
    ph, pw = h - a.shape[0], w - a.shape[1]
    if ph > 0 or pw > 0:
        a = np.pad(a, ((0, max(ph, 0)), (0, max(pw, 0))), mode="edge")
    return a


def shift_bilinear(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate content by (+dx, +dy) pixels with bilinear weights and zero fill."""
    if dx == 0 and dy == 0:
        return a
    ix, iy = math.floor(dx), math.floor(dy)
    fx, fy = dx - ix, dy - iy
    out = np.zeros_like(a)
    for sy, wy in ((iy, 1.0 - fy), (iy + 1, fy)):
        if wy <= 1e-7:
            continue
        for sx, wx in ((ix, 1.0 - fx), (ix + 1, fx)):
            wgt = wy * wx
            if wgt <= 1e-7:
                continue
            _add_shifted(out, a, sx, sy, np.float32(wgt))
    return out


def _add_shifted(out, a, sx, sy, w):
    h, wd = a.shape
    if abs(sx) >= wd or abs(sy) >= h:
        return
    ys_dst = slice(max(sy, 0), h + min(sy, 0))
    ys_src = slice(max(-sy, 0), h - max(sy, 0))
    xs_dst = slice(max(sx, 0), wd + min(sx, 0))
    xs_src = slice(max(-sx, 0), wd - max(sx, 0))
    out[ys_dst, xs_dst] += w * a[ys_src, xs_src]


def world_normal_noise(ix0: int, iy0: int, ny: int, nx: int, seed: int, salt: int = 0) -> np.ndarray:
    """Standard-normal float32 noise on world cells (iy0..iy0+ny, ix0..ix0+nx).

    Cells are grouped in 128x128 world tiles, each drawn from a PCG64 stream
    seeded by ``hash_seed(seed, TEXTURE_KIND, mix_cell(tx, ty), salt)``, so the
    value of a cell never depends on the view it is rendered in.
    """
    out = np.empty((ny, nx), np.float32)
    tx0, tx1 = ix0 // _TILE, (ix0 + nx - 1) // _TILE
    ty0, ty1 = iy0 // _TILE, (iy0 + ny - 1) // _TILE
    for ty in range(ty0, ty1 + 1):
        ya = max(iy0, ty * _TILE)
        yb = min(iy0 + ny, (ty + 1) * _TILE)
        for tx in range(tx0, tx1 + 1):
            xa = max(ix0, tx * _TILE)
            xb = min(ix0 + nx, (tx + 1) * _TILE)
            rng = np.random.Generator(np.random.PCG64(
                hash_seed(seed, TEXTURE_KIND, mix_cell(tx, ty), salt)))
            tile = rng.standard_normal((_TILE, _TILE), dtype=np.float32)
            out[ya - iy0:yb - iy0, xa - ix0:xb - ix0] = \
                tile[ya - ty * _TILE:yb - ty * _TILE, xa - tx * _TILE:xb - tx * _TILE]
    return out


def disk_profile(shape: tuple[int, int], cx: float, cy: float, radius: float,
                 edge_sigma: float) -> np.ndarray | None:
    """Soft top-hat (1 inside, 0 outside) or ``None`` when it covers the whole frame."""
    h, w = shape
    far = math.hypot(max(cx + 0.5, w - cx - 0.5), max(cy + 0.5, h - cy - 0.5))
    if radius - 4.0 * edge_sigma >= far:
        return None
    from scipy.special import erfc
    y = (np.arange(h, dtype=np.float32) - np.float32(cy))[:, None]
    x = (np.arange(w, dtype=np.float32) - np.float32(cx))[None, :]
    r = np.sqrt(x * x + y * y)
    return (0.5 * erfc((r - radius) / (math.sqrt(2.0) * max(edge_sigma, 1e-3)))).astype(np.float32)


def chi_and_gradient(ab, kx, ky, wavelength_nm: float, gradient: bool = True, dtype=np.float64):
    """(chi, dchi/dkx, dchi/dky) of an :class:`~de_twin.optics.aberrations.Aberrations` set
    (rad and rad nm; kx, ky in 1/nm, broadcastable).

    Same values as ``ab.chi`` / ``ab.gradient``. Sets made only of C1, A1, C3 and C5 (the
    usual uncorrected / probe-corrected column) take a real-arithmetic fast path in ``dtype``;
    anything else builds the powers of ``w`` by repeated complex multiplication."""
    from ..optics.aberrations import TERMS

    lam = float(wavelength_nm)
    s = 2.0 * math.pi / lam
    if set(ab.coeffs) <= {"C1", "A1", "C3", "C5"}:
        f = np.dtype(dtype).type
        kx = np.asarray(kx, dtype)
        ky = np.asarray(ky, dtype)
        c1, c3, c5 = (ab[n].real for n in ("C1", "C3", "C5"))
        a1 = ab["A1"]
        l2 = lam * lam
        k2 = kx * kx + ky * ky
        # chi = s [ l2/2 (C1 k2 + a (kx2 - ky2) + 2 b kx ky) + C3 l2^2 k2^2 / 4 + C5 l2^3 k2^3 / 6 ]
        radial = f(s * l2 * c1 / 2)
        if c3 or c5:
            poly = f(s * l2 * l2 * c3 / 4) + f(s * l2 ** 3 * c5 / 6) * k2
            chi = k2 * (radial + k2 * poly)
        else:
            chi = k2 * radial
        if a1:
            chi = chi + f(s * l2 * a1.real / 2) * (kx * kx - ky * ky) + f(s * l2 * a1.imag) * (kx * ky)
        if not gradient:
            return chi, None, None
        # grad = s [ l2 (C1 k + (a kx + b ky, -a ky + b kx)) + C3 l2^2 k2 k + C5 l2^3 k2^2 k ]
        rad = f(s * l2 * c1) + k2 * (f(s * l2 * l2 * c3) + f(s * l2 ** 3 * c5) * k2)
        gx = rad * kx
        gy = rad * ky
        if a1:
            ar, ai = f(s * l2 * a1.real), f(s * l2 * a1.imag)
            gx = gx + ar * kx + ai * ky
            gy = gy - ar * ky + ai * kx
        return chi, gx, gy
    w = lam * (np.asarray(kx, np.float64) + 1j * np.asarray(ky, np.float64))
    wb = np.conj(w)
    need = max([max(TERMS[n][0], TERMS[n][1]) for n in ab.coeffs] or [0])
    pw, pwb = [np.ones_like(w)], [np.ones_like(wb)]
    for _ in range(need):
        pw.append(pw[-1] * w)
        pwb.append(pwb[-1] * wb)
    acc = np.zeros(np.broadcast(w, wb).shape, np.complex128)
    ga = np.zeros_like(acc) if gradient else None
    gb = np.zeros_like(acc) if gradient else None
    for name, c in ab.coeffs.items():
        p, q, pre, _ = TERMS[name]
        acc += (pre * c) * (pwb[p] * pw[q])
        if gradient:
            if p:
                ga += (pre * c * p) * (pwb[p - 1] * pw[q])
            if q:
                gb += (pre * c * q) * (pwb[p] * pw[q - 1])
    chi = (s * acc.real).astype(dtype, copy=False)
    if not gradient:
        return chi, None, None
    # dw/dkx = dwb/dkx = lam; dw/dky = i lam, dwb/dky = -i lam
    return (chi, (2.0 * math.pi * (ga + gb).real).astype(dtype, copy=False),
            (2.0 * math.pi * (-1j * ga + 1j * gb).real).astype(dtype, copy=False))
