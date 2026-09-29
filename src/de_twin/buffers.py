"""Recycled large arrays.

A new large NumPy array costs the page faults of its first touch, and on Windows these are
slow when the first touch is a multithreaded kernel (a 14 MB complex64 spectrum written by a
threaded FFT: ~17 ms of faults for ~4 ms of work). The render allocates the same few shapes
for every new view (field-map layers, exit-wave spectra, raster images), so :func:`empty`
hands back an array of the same shape and dtype that nothing references any more (its pages
still mapped) instead of a new one. An array is free when only this pool holds it; views keep
their base alive, so a live view is never recycled. The pool keeps at most ``MAX_BYTES`` of
unreferenced arrays.
"""

from __future__ import annotations

import sys
import threading

import numpy as np

MAX_BYTES = 768 << 20
MIN_BYTES = 1 << 20  # smaller arrays: plain np.empty

_POOL: list = []
_LOCK = threading.Lock()


def _free(a) -> bool:
    # references: the pool list, the caller's loop variable, getrefcount's argument
    return sys.getrefcount(a) <= 3


def empty(shape, dtype=np.float64) -> np.ndarray:
    """``np.empty(shape, dtype)``, recycled when a free array of that shape and dtype exists
    (uninitialised either way)."""
    shape = tuple(int(v) for v in (shape if np.ndim(shape) else (shape,)))
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if nbytes < MIN_BYTES:
        return np.empty(shape, dtype)
    with _LOCK:
        for i, a in enumerate(_POOL):
            if a.shape == shape and a.dtype == dtype and _free(a):
                _POOL.append(_POOL.pop(i))  # most recently used last
                return a
        a = np.empty(shape, dtype)
        a.reshape(-1).view(np.uint8)[::4096] = 0  # fault the pages in here, single-threaded
        _POOL.append(a)
        total = sum(x.nbytes for x in _POOL if _free(x))
        i = 0
        while total > MAX_BYTES and i < len(_POOL):
            x = _POOL[i]
            if _free(x) and x is not a:
                total -= x.nbytes
                del _POOL[i]
            else:
                i += 1
        return a


def zeros(shape, dtype=np.float64) -> np.ndarray:
    a = empty(shape, dtype)
    a.fill(0)
    return a


def full(shape, value, dtype) -> np.ndarray:
    a = empty(shape, dtype)
    a.fill(value)
    return a
