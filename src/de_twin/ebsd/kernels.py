"""numba kernels for EBSD: the kinematic master-pattern sum and the per-pattern projection.

Square Lambert projection (Rosca-Lambert, as EMsoft stores master patterns) of the northern
hemisphere ``z >= 0`` onto ``[-L, L]^2``, ``L = sqrt(pi / 2)``, sampled at ``N x N`` nodes
(``linspace(-L, L, N)``, so the equator is the square's edge). Kinematic Kikuchi patterns are
centrosymmetric (``|F_g| = |F_-g|`` and every band is symmetric about its plane), so a
direction with ``z < 0`` reads the master at ``-direction``: one hemisphere holds the sphere.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

SQRT_PI = math.sqrt(math.pi)
LAMBERT_L = math.sqrt(math.pi / 2.0)

#: Band profile: an excess line just inside each edge and a deficiency just outside, at
#: ``EDGE_OFFSET`` band half-widths from it (dynamical-looking edges on a kinematic band).
EDGE_EXCESS = 0.25
EDGE_DEFICIENCY = 0.35
EDGE_OFFSET = 0.08
EDGE_MIN_SIGMA = 0.05


def _njit(**kw):
    return nb.njit(cache=True, nogil=True, fastmath=True, error_model="numpy", **kw)


# ---------------------------------------------------------------------- Lambert (numpy)
def lambert_directions(n: int) -> np.ndarray:
    """(n, n, 3) unit directions at the nodes of the square-Lambert northern hemisphere, row =
    Lambert Y, column = Lambert X."""
    L = LAMBERT_L
    c = np.linspace(-L, L, n)
    X, Y = np.meshgrid(c, c)
    out = np.empty((n, n, 3))
    a = np.abs(Y) <= np.abs(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        # |Y| <= |X|
        q = 2.0 * X / math.pi * np.sqrt(np.maximum(math.pi - X * X, 0.0))
        ang = np.where(X != 0, math.pi * Y / (4.0 * np.where(X != 0, X, 1.0)), 0.0)
        xa, ya, za = q * np.cos(ang), q * np.sin(ang), 1.0 - 2.0 * X * X / math.pi
        # |X| < |Y|
        q = 2.0 * Y / math.pi * np.sqrt(np.maximum(math.pi - Y * Y, 0.0))
        ang = np.where(Y != 0, math.pi * X / (4.0 * np.where(Y != 0, Y, 1.0)), 0.0)
        xb, yb, zb = q * np.sin(ang), q * np.cos(ang), 1.0 - 2.0 * Y * Y / math.pi
    out[..., 0] = np.where(a, xa, xb)
    out[..., 1] = np.where(a, ya, yb)
    out[..., 2] = np.where(a, za, zb)
    out[(X == 0) & (Y == 0)] = (0.0, 0.0, 1.0)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def lambert_xy(v) -> tuple[np.ndarray, np.ndarray]:
    """Square-Lambert (X, Y) of unit directions (..., 3) (folded to z >= 0)."""
    v = np.asarray(v, np.float64)
    v = np.where(v[..., 2:3] < 0, -v, v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    q = np.sqrt(np.maximum(2.0 * (1.0 - z), 0.0))
    a = np.abs(y) <= np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = np.where(x >= 0, 1.0, -1.0)
        sy = np.where(y >= 0, 1.0, -1.0)
        Xa = sx * q * SQRT_PI / 2.0
        Ya = sx * q * 2.0 / SQRT_PI * np.arctan(np.where(x != 0, y / np.where(x != 0, x, 1.0), 0.0))
        Xb = sy * q * 2.0 / SQRT_PI * np.arctan(np.where(y != 0, x / np.where(y != 0, y, 1.0), 0.0))
        Yb = sy * q * SQRT_PI / 2.0
    return np.where(a, Xa, Xb), np.where(a, Ya, Yb)


# ---------------------------------------------------------------------- master sum
@_njit(parallel=True)
def accumulate_bands(dirs, normals, half_width, amp, sigma, sigma_edge, reach, out):
    """``out[p] = sum_k amp_k * P(|dirs_p . normals_k| / half_width_k)``: every band ``k`` a
    smoothed top hat of half-width ``sin(theta_B)`` (edge blur ``sigma``, in half-widths) with
    an excess line just inside each edge and a deficiency just outside (``sigma_edge``).
    ``reach``: |t| beyond which band k contributes nothing (skipped)."""
    P = dirs.shape[0]
    K = normals.shape[0]
    r2 = 1.0 / math.sqrt(2.0)
    for p in nb.prange(P):
        x, y, z = dirs[p, 0], dirs[p, 1], dirs[p, 2]
        acc = 0.0
        for k in range(K):
            u = abs(x * normals[k, 0] + y * normals[k, 1] + z * normals[k, 2])
            t = u / half_width[k]
            if t > reach[k]:
                continue
            s = sigma[k]
            band = 0.5 * (math.erf((1.0 - t) * r2 / s) + math.erf((1.0 + t) * r2 / s))
            se = sigma_edge[k]
            a = t - 1.0 + EDGE_OFFSET
            b = t - 1.0 - EDGE_OFFSET
            edge = EDGE_EXCESS * math.exp(-0.5 * a * a / (se * se)) \
                - EDGE_DEFICIENCY * math.exp(-0.5 * b * b / (se * se))
            acc += amp[k] * (band + edge)
        out[p] = acc


# ---------------------------------------------------------------------- projection
@_njit()
def _lookup(master, scale, cx, cy, cz):
    """Bilinear master code at crystal direction (cx, cy, cz) (unit; folded to z >= 0)."""
    if cz < 0.0:
        cx, cy, cz = -cx, -cy, -cz
    q = math.sqrt(max(2.0 * (1.0 - cz), 0.0))
    if abs(cy) <= abs(cx):
        if cx == 0.0:
            X, Y = 0.0, 0.0
        else:
            sg = 1.0 if cx > 0.0 else -1.0
            X = sg * q * (SQRT_PI * 0.5)
            Y = sg * q * (2.0 / SQRT_PI) * math.atan(cy / cx)
    else:
        sg = 1.0 if cy > 0.0 else -1.0
        X = sg * q * (2.0 / SQRT_PI) * math.atan(cx / cy)
        Y = sg * q * (SQRT_PI * 0.5)
    n = master.shape[0]
    fx = (X + LAMBERT_L) * scale
    fy = (Y + LAMBERT_L) * scale
    ix = int(fx)
    iy = int(fy)
    if ix < 0:
        ix = 0
    elif ix > n - 2:
        ix = n - 2
    if iy < 0:
        iy = 0
    elif iy > n - 2:
        iy = n - 2
    tx = min(max(fx - ix, 0.0), 1.0)
    ty = min(max(fy - iy, 0.0), 1.0)
    v00 = np.float32(master[iy, ix])
    v01 = np.float32(master[iy, ix + 1])
    v10 = np.float32(master[iy + 1, ix])
    v11 = np.float32(master[iy + 1, ix + 1])
    return (v00 * (1.0 - tx) + v01 * tx) * (1.0 - ty) + (v10 * (1.0 - tx) + v11 * tx) * ty


@_njit(parallel=True)
def project(master1, a1, b1, m1, master2, a2, b2, m2, w2, x0, y0, dd, ox, oy, bx, by, raw,
            normal, fwd, i_cos, i_lobe, kappa, contrast, out):
    """One EBSD pattern on a (binned) detector grid.

    Pixel (i, j) of ``out`` is the sensor area ``[ox + j bx, ox + (j + 1) bx) x [oy + i by, ...)``;
    its direction from the source is ``(X, Y, dd) / |.|`` with ``X``, ``Y`` its centre minus the
    pattern centre ``(x0, y0)``, all in sensor pixels. ``m1`` / ``m2``: detector -> crystal
    matrices of the grain (and of the neighbour mixed in with weight ``w2``); ``master*`` the
    uint8 master codes, value = ``a + b * code``.

    ``raw``: ``out`` is the blended master value (no background, no solid angle).
    Otherwise ``out`` is electrons per sensor pixel per second::

        (i_cos (r.n)/pi + i_lobe vMF(r; fwd, kappa)) * dOmega * (1 + contrast * m)

    with ``dOmega = cos^3(gamma) / dd^2`` the solid angle of one sensor pixel, zero where the
    direction is below the sample surface (a soft shadow edge).
    """
    h, w = out.shape
    s1 = (master1.shape[0] - 1) / (2.0 * LAMBERT_L)
    s2 = (master2.shape[0] - 1) / (2.0 * LAMBERT_L)
    w1 = 1.0 - w2
    nx_, ny_, nz_ = normal[0], normal[1], normal[2]
    fx_, fy_, fz_ = fwd[0], fwd[1], fwd[2]
    vmf = kappa / (2.0 * math.pi * (1.0 - math.exp(-2.0 * kappa))) if kappa > 0 else 1.0 / (4.0 * math.pi)
    inv_dd2 = 1.0 / (dd * dd)
    for i in nb.prange(h):
        Y = oy + (i + 0.5) * by - y0
        for j in range(w):
            X = ox + (j + 0.5) * bx - x0
            inv = 1.0 / math.sqrt(X * X + Y * Y + dd * dd)
            rx, ry, rz = X * inv, Y * inv, dd * inv
            c0 = m1[0, 0] * rx + m1[0, 1] * ry + m1[0, 2] * rz
            c1 = m1[1, 0] * rx + m1[1, 1] * ry + m1[1, 2] * rz
            c2 = m1[2, 0] * rx + m1[2, 1] * ry + m1[2, 2] * rz
            val = w1 * (a1 + b1 * _lookup(master1, s1, c0, c1, c2))
            if w2 > 0.0:
                c0 = m2[0, 0] * rx + m2[0, 1] * ry + m2[0, 2] * rz
                c1 = m2[1, 0] * rx + m2[1, 1] * ry + m2[1, 2] * rz
                c2 = m2[2, 0] * rx + m2[2, 1] * ry + m2[2, 2] * rz
                val += w2 * (a2 + b2 * _lookup(master2, s2, c0, c1, c2))
            if raw:
                out[i, j] = val
                continue
            rn = rx * nx_ + ry * ny_ + rz * nz_
            if rn <= 0.0:
                out[i, j] = 0.0
                continue
            shadow = min(rn / 0.03, 1.0)
            rf = rx * fx_ + ry * fy_ + rz * fz_
            bg = i_cos * rn / math.pi + i_lobe * vmf * math.exp(kappa * (rf - 1.0))
            domega = rz * rz * rz * inv_dd2
            v = shadow * bg * domega * (1.0 + contrast * val)
            out[i, j] = v if v > 0.0 else 0.0
