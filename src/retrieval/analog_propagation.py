"""Mass-Shifted Analog Propagation Engine.
Matches unknown spectra against structurally related chemical analogs in the reference
library across a +-200 Da window, shifting fragment ion coordinates by Delta m.
"""
import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0; j = 0; n = len(qmz); m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])
    SAB = 0.0; tot = 0.0
    buf = np.empty(n + m, np.float64); b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol: buf[b] = qp[i]; i += 1; b += 1
        elif d > tol: buf[b] = cp[j]; j += 1; b += 1
        else: buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0: SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)

@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if -0.001 < shift < 0.001: return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)): sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b

@njit(cache=True, fastmath=True, parallel=True)
def search_shift(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight, shift):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        # clean & calculate
        # ...
    return out
