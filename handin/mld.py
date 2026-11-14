"""Near-exact maximum-likelihood decoding by a pruned dynamic programme over the error mechanisms, from scratch.

The optimal prediction for each logical bit is its posterior P(O_j = 1 | D): the sum over all error vectors e with
H e = D of prod p_i^e_i (1 - p_i)^(1 - e_i), split by L e. We compute that sum by sweeping the error mechanisms in
order of their first detector.

State. A state is a partial error vector, summarised by which detectors it still disagrees with (H e XOR D, 70 bits)
and its logical flips (L e, 4 bits), packed in two uint64 words. Its weight is the total probability, relative to
the no-error vector, of all partial error vectors with that summary. The sweep starts from the empty error vector,
which disagrees with every fired detector.

Step. Mechanism k maps every state s to s (weight w, the mechanism did not fire) and s XOR col_k (weight
w * p_k / (1 - p_k)); equal states are merged by adding their weights (hash table). Once the last mechanism that
touches a detector has been applied, states that still disagree with D on it can never be fixed and are dropped.
At the end every detector agrees, so the surviving states differ only in their logical bits, and their weights are
the posterior of the 16 logical outcomes.

Pruning (a beam search). Exactly, the number of states would be 2^(frontier width + 4). A state is scored by its
log weight plus a cost-to-go estimate: each detector it still disagrees with is charged hexp * log(p/(1-p)) of the
most likely remaining mechanism that touches that detector. A typical mechanism fixes two detectors, hence hexp ~
0.5. Without this term the no-error state would look best until the fired detectors close, and the paths that
explain them would be pruned. States scoring below best + log(eps) are dropped, and at most max_states are kept.

Speed. Every state is its own "did not fire" child, so states are updated in place; only the flipped children that
pass the threshold (about 1% per state and step) are looked up in the hash table and merged or appended. The change
of a child's cost-to-go is linear in the state's disagreeing detectors within the column, so it is a sum over a
couple of bits of a precomputed table; dropped states are removed in periodic compactions.
"""
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numba
from numba import njit

_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_CTZ_TABLE = np.zeros(64, np.int64)
with np.errstate(over="ignore"):  # the multiplication is meant to wrap around
    for _i in range(64):
        _CTZ_TABLE[((np.uint64(1) << np.uint64(_i)) * _DEBRUIJN) >> np.uint64(58)] = _i


@njit(cache=True)
def _ctz(x):
    """Index of the lowest set bit of a non-zero uint64 (de Bruijn multiplication)."""
    return _CTZ_TABLE[((x & (~x + np.uint64(1))) * _DEBRUIJN) >> np.uint64(58)]


VEC_MIN = 24  # state count from which a step uses the vectorisable passes


def build_mld(H: np.ndarray, L: np.ndarray, p: np.ndarray, hexp=0.5):
    """Precompute the sweep order and the per-step tables the kernel needs from the error model."""
    H = np.asarray(H, np.uint8)
    L = np.asarray(L, np.uint8)
    p = np.asarray(p, np.float64)
    nd, n = H.shape
    nobs = L.shape[0]
    assert 64 < nd and nd + nobs <= 128
    rows = np.arange(nd)[:, None]
    first = np.where(H, rows, nd).min(0)
    last = np.where(H, rows, -1).max(0)
    order = np.lexsort((last, first))
    Ho = H[:, order].astype(bool)
    Lo = L[:, order].astype(bool)
    full = np.concatenate([Ho, Lo], 0)  # (nd + nobs, n): detector bits then logical bits
    w0 = np.uint64(1) << np.arange(64, dtype=np.uint64)
    w1 = np.uint64(1) << np.arange(full.shape[0] - 64, dtype=np.uint64)

    def pack(bits):  # (rows, n) bool -> two uint64 words per column
        lo = (bits[:64] * w0[: min(64, bits.shape[0]), None]).sum(0, dtype=np.uint64)
        hi = (bits[64:] * w1[: bits.shape[0] - 64, None]).sum(0, dtype=np.uint64)
        return lo.astype(np.uint64), hi.astype(np.uint64)

    c0, c1 = pack(full)  # the column, including its logical flips
    m0, m1 = pack(Ho)  # its detectors only
    lr = np.log(p[order] / (1 - p[order]))
    k_idx = np.arange(n)
    lastuse = np.where(Ho, k_idx[None, :], -1).max(1)  # step after which each detector is closed
    cl0, cl1 = pack(Ho & (lastuse[:, None] == k_idx[None, :]))  # detectors closed by step k
    # hcost[k, d]: cost-to-go of a disagreeing detector d before step k (row n: after the last step) = hexp * the
    # largest log ratio among mechanisms k.. touching d, or 0 if there are none left
    share = np.where(Ho, (lr * hexp)[None, :], -np.inf)  # (nd, n)
    suffix = np.maximum.accumulate(share[:, ::-1], axis=1)[:, ::-1]
    hcost = np.zeros((n + 1, nd))
    hcost[:n] = np.where(np.isfinite(suffix.T), suffix.T, 0.0)
    before = np.where(Ho.T, hcost[:n], 0.0)  # (n, nd), only on the column's detectors
    after = np.where(Ho.T, hcost[1:], 0.0)
    v0 = after - before  # change of a disagreeing detector's charge when the mechanism does not fire
    v1 = -before - after  # extra change for the flipped child when the detector disagrees (it becomes fixed)
    d1z = after.sum(1)  # flipped child of a state agreeing on the whole column: every detector becomes disagreeing
    hasv0 = (v0 != 0).any(1)
    wmax = int(Ho.sum(0).max())
    cdet = np.zeros((n, wmax), np.int64)  # the column's detectors, padded with a dummy entry whose values are 0
    cv0 = np.zeros((n, wmax))
    cv1 = np.zeros((n, wmax))
    cw = Ho.sum(0).astype(np.int64)
    for k in range(n):
        dets = np.flatnonzero(Ho[:, k])
        cdet[k, : len(dets)] = dets
        cv0[k, : len(dets)] = v0[k, dets]
        cv1[k, : len(dets)] = v1[k, dets]
    return dict(c0=c0, c1=c1, m0=m0, m1=m1, cl0=cl0, cl1=cl1, lr=lr, v0=v0, v1=v1, cdet=cdet, cv0=cv0, cv1=cv1, cw=cw, d1z=d1z,
                hasv0=hasv0, h0=hcost[0].copy(), nd=nd, nobs=nobs)


@njit(cache=True, inline="always")
def _hash(a, b, mask):
    h = (a * np.uint64(0x9E3779B97F4A7C15)) ^ (b * np.uint64(0xC2B2AE3D27D4EB4F))
    h ^= h >> np.uint64(29)
    return np.int64(h & mask)


@njit(cache=True, inline="always")
def _insert(table, tmask, a, b, idx):
    h = _hash(a, b, tmask)
    while table[h] >= 0:
        h = np.int64((np.uint64(h) + np.uint64(1)) & tmask)
    table[h] = idx


@njit(cache=True, inline="always")
def _find(table, tmask, ka, kb, a, b):
    h = _hash(a, b, tmask)
    while True:
        t = table[h]
        if t == -1:  # empty; -2 marks a removed state, whose slot still continues probe sequences
            return -1
        if t >= 0 and ka[t] == a and kb[t] == b:
            return t
        h = np.int64((np.uint64(h) + np.uint64(1)) & tmask)


@njit(cache=True, inline="always")
def _logadd(x, y):
    if x < y:
        x, y = y, x
    if y == -np.inf:
        return x
    return x + np.log1p(np.exp(y - x))



# General-purpose sorting adaptation from numba==0.68.1, numba/misc/quicksort.py.
# This is sorting code, not an external QEC decoder.
# Copyright (c) 2012, Anaconda, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@njit(cache=True)
def _argsort_states(scores, keep):
    """Exact prefix sorter adapted from Numba's general-purpose quicksort (BSD-2-Clause).

    Move keys with indices instead of looking them up indirectly. The pivot,
    swaps and insertion-sort cutoff match numba==0.68.1 exactly. Subpartitions
    wholly beyond the returned prefix need no further sorting.
    """
    # Map IEEE float64 keys to an unsigned order with exactly the same
    # comparisons as Numba's float argsort, including NaNs and signed zeros.
    values = (-scores).view(np.uint64)
    sign = np.uint64(0x8000000000000000)
    magnitude_mask = np.uint64(0x7fffffffffffffff)
    infinity = np.uint64(0x7ff0000000000000)
    for i in range(len(values)):
        bits = values[i]
        magnitude = bits & magnitude_mask
        if magnitude == 0:
            values[i] = sign
        elif magnitude > infinity:
            values[i] = np.uint64(0xffffffffffffffff)
        else:
            values[i] = ~bits if bits & sign else bits ^ sign
    n = len(values)
    indices = np.arange(n)
    if n < 2 or keep == 0:
        return indices[:keep]
    lows = np.empty(100, np.int64)
    highs = np.empty(100, np.int64)
    lows[0], highs[0] = 0, n-1
    count = 1
    while count:
        count -= 1
        low, high = lows[count], highs[count]
        if low >= keep:
            continue
        while high-low >= 15:
            if low >= keep:
                break
            mid = (low+high) >> 1
            if values[mid] < values[low]:
                values[low], values[mid] = values[mid], values[low]
                indices[low], indices[mid] = indices[mid], indices[low]
            if values[high] < values[mid]:
                values[high], values[mid] = values[mid], values[high]
                indices[high], indices[mid] = indices[mid], indices[high]
            if values[mid] < values[low]:
                values[low], values[mid] = values[mid], values[low]
                indices[low], indices[mid] = indices[mid], indices[low]
            pivot = values[mid]
            values[high], values[mid] = values[mid], values[high]
            indices[high], indices[mid] = indices[mid], indices[high]
            i, j = low, high-1
            while True:
                # values[high] is the pivot and values[low] <= pivot throughout, so both scans stop inside
                # [low, high] without the bounds tests of the original
                while values[i] < pivot:
                    i += 1
                while pivot < values[j]:
                    j -= 1
                if i >= j:
                    break
                values[i], values[j] = values[j], values[i]
                indices[i], indices[j] = indices[j], indices[i]
                i += 1
                j -= 1
            values[i], values[high] = values[high], values[i]
            indices[i], indices[high] = indices[high], indices[i]
            assert count < 100
            if high-i > i-low:
                if high > i:
                    lows[count], highs[count] = i+1, high
                    count += 1
                high = i-1
            else:
                if i > low:
                    lows[count], highs[count] = low, i-1
                    count += 1
                low = i+1
        if low >= keep:
            continue
        for i in range(low+1, high+1):
            key, index = values[i], indices[i]
            j = i
            while j > low and key < values[j-1]:
                values[j] = values[j-1]
                indices[j] = indices[j-1]
                j -= 1
            values[j], indices[j] = key, index
    return indices[:keep]


@njit(cache=True, fastmath={"nsz", "reassoc", "contract", "arcp"})
def _mld_shot(c0, c1, m0, m1, cl0, cl1, lr, v0, v1, hasv0, cdet, cv0, cv1, cw, d1z, h0, D0, D1, nd, leps, max_states, compact_every,
              ka, kb, lw, sc, cka, ckb, clw, csc, cs, ce, table, out16, stats):
    """Posterior weights of the 16 logical outcomes for one shot, written to out16. The remaining arrays are
    scratch space of size >= 2 * max_states + 1 (table: a power of two >= 4 times that). The weights in out16 are
    relative to the largest one, whose log is written to stats[1]; stats[0] is the peak state count."""
    cur = 64  # table[:cur] is in use, sized to the state count
    tmask = np.uint64(cur - 1)
    table[:cur] = -1
    h = 0.0
    for d in range(64):
        if (D0 >> np.uint64(d)) & np.uint64(1):
            h += h0[d]
    for d in range(64, nd):
        if (D1 >> np.uint64(d - 64)) & np.uint64(1):
            h += h0[d]
    ka[0] = D0
    kb[0] = D1
    lw[0] = 0.0
    sc[0] = h
    _insert(table, tmask, D0, D1, 0)
    occ = 1  # upper bound on the non-empty table slots (states and removed-state markers)
    ns = 1
    peak = 1
    best = h
    since = 0
    ninf = -np.inf
    for k in range(c0.shape[0]):
        x0 = c0[k]
        x1 = c1[k]
        q0 = cl0[k]
        q1 = cl1[k]
        closes = (q0 | q1) != 0
        lrk = lr[k]
        dz = d1z[k]
        thr = best + leps
        if ns < VEC_MIN:  # few states: one branchy pass per state is cheapest
            mk0 = m0[k]
            mk1 = m1[k]
            hv = hasv0[k]
            thrz = thr - lrk - dz  # a state agreeing on the whole column has a passing flipped child iff sc >= thrz
            nbest = ninf
            nc = 0
            for i in range(ns):
                s = sc[i]
                a = ka[i]
                b = kb[i]
                y0 = a & mk0
                y1 = b & mk1
                if (y0 | y1) == 0:
                    if s > nbest:
                        nbest = s
                    if s >= thrz and not closes:
                        cka[nc] = a ^ x0
                        ckb[nc] = b ^ x1
                        clw[nc] = lw[i] + lrk
                        csc[nc] = s + lrk + dz
                        nc += 1
                    continue
                if s == ninf:
                    continue
                e1 = dz
                e0 = 0.0
                while y0:
                    d = _ctz(y0)
                    e1 += v1[k, d]
                    if hv:
                        e0 += v0[k, d]
                    y0 &= y0 - np.uint64(1)
                while y1:
                    d = 64 + _ctz(y1)
                    e1 += v1[k, d]
                    if hv:
                        e0 += v0[k, d]
                    y1 &= y1 - np.uint64(1)
                if closes:
                    bad0 = ((a & q0) | (b & q1)) != 0  # still disagrees on a detector that closes now
                    bad1 = ((~a & q0) | (~b & q1)) != 0  # the flipped child would
                else:
                    bad0 = False
                    bad1 = False
                if not bad1:
                    c = s + lrk + e1
                    if c >= thr:
                        cka[nc] = a ^ x0
                        ckb[nc] = b ^ x1
                        clw[nc] = lw[i] + lrk
                        csc[nc] = c
                        nc += 1
                if bad0:
                    sc[i] = ninf
                else:
                    s += e0
                    sc[i] = s
                    if s > nbest:
                        nbest = s
        else:
            # vectorisable passes over the states: cost-to-go changes from the column's disagreeing detectors
            for i in range(ns):
                cs[i] = dz
                ce[i] = 0.0
            if hasv0[k]:
                for j in range(cw[k]):
                    d = cdet[k, j]
                    f1 = cv1[k, j]
                    f0 = cv0[k, j]
                    if d < 64:
                        sh = np.uint64(d)
                        for i in range(ns):
                            bit = np.float64((ka[i] >> sh) & np.uint64(1))
                            cs[i] += f1 * bit
                            ce[i] += f0 * bit
                    else:
                        sh = np.uint64(d - 64)
                        for i in range(ns):
                            bit = np.float64((kb[i] >> sh) & np.uint64(1))
                            cs[i] += f1 * bit
                            ce[i] += f0 * bit
            else:
                # Every cv0 term is exactly zero, so ce remains its initialized
                # zero. The cs updates keep the same order and operations.
                for j in range(cw[k]):
                    d = cdet[k, j]
                    f1 = cv1[k, j]
                    if d < 64:
                        sh = np.uint64(d)
                        for i in range(ns):
                            bit = np.float64((ka[i] >> sh) & np.uint64(1))
                            cs[i] += f1 * bit
                    else:
                        sh = np.uint64(d - 64)
                        for i in range(ns):
                            bit = np.float64((kb[i] >> sh) & np.uint64(1))
                            cs[i] += f1 * bit
            nbest = ninf
            if closes:
                for i in range(ns):
                    a = ka[i]
                    b = kb[i]
                    s = sc[i]
                    bad1 = ((~a & q0) | (~b & q1)) != 0
                    cs[i] = ninf if bad1 else s + lrk + cs[i]
                    s = ninf if ((a & q0) | (b & q1)) != 0 else s + ce[i]
                    sc[i] = s
                    nbest = max(nbest, s)
            else:
                for i in range(ns):
                    cs[i] += sc[i] + lrk
                    s = sc[i] + ce[i]
                    sc[i] = s
                    nbest = max(nbest, s)
            nc = 0
            for i in range(ns):
                if cs[i] >= thr:
                    cka[nc] = ka[i] ^ x0
                    ckb[nc] = kb[i] ^ x1
                    clw[nc] = lw[i] + lrk
                    csc[nc] = cs[i]
                    nc += 1
        # merge the flipped children (their weights were taken before this step's updates)
        for c in range(nc):
            a = cka[c]
            b = ckb[c]
            t = _find(table, tmask, ka, kb, a, b)
            if t >= 0:
                l = _logadd(lw[t], clw[c])
                sc[t] += l - lw[t]
                lw[t] = l
            else:
                t = ns
                ka[t] = a
                kb[t] = b
                lw[t] = clw[c]
                sc[t] = csc[c]
                ns += 1
                occ += 1
                if 2 * occ > cur:  # keep the load factor, removed-state markers included, below 1/2
                    cur = 64
                    while cur < 4 * ns:
                        cur *= 2
                    tmask = np.uint64(cur - 1)
                    table[:cur] = -1
                    for q in range(ns):
                        _insert(table, tmask, ka[q], kb[q], q)
                    occ = ns
                else:
                    _insert(table, tmask, a, b, t)
            if sc[t] > nbest:
                nbest = sc[t]
        best = nbest
        since += 1
        if ns > max_states or since >= compact_every or closes:
            since = 0
            thr = best + leps
            pos = ce.view(np.int64)  # pos[old index] = new index or -1; ce is free until the next step
            m = 0
            for i in range(ns):
                if sc[i] >= thr:
                    ka[m] = ka[i]
                    kb[m] = kb[i]
                    lw[m] = lw[i]
                    sc[m] = sc[i]
                    pos[i] = m
                    m += 1
                else:
                    pos[i] = -1
            nold = ns
            ns = m
            if ns == 0:
                break
            if ns > max_states:
                idx = _argsort_states(sc[:ns], max_states)
                # Gather all fields before copying back; preserve the exact
                # original state order without four temporary indexed arrays.
                for j in range(max_states):
                    source_index = idx[j]
                    cka[j] = ka[source_index]
                    ckb[j] = kb[source_index]
                    clw[j] = lw[source_index]
                    csc[j] = sc[source_index]
                ka[:max_states] = cka[:max_states]
                kb[:max_states] = ckb[:max_states]
                lw[:max_states] = clw[:max_states]
                sc[:max_states] = csc[:max_states]
                inv = cs.view(np.int64)  # inv[index before sorting] = index after, or -1
                for j in range(m):
                    inv[j] = -1
                for j in range(max_states):
                    inv[idx[j]] = j
                for i in range(nold):
                    if pos[i] >= 0:
                        pos[i] = inv[pos[i]]
                ns = max_states
            if 2 * occ <= cur and cur <= 16 * ns:
                # the states keep their keys, so every slot keeps its place on its probe sequence: renumber the
                # surviving states and mark the removed ones, instead of inserting every state again
                for h in range(cur):
                    t = table[h]
                    if t >= 0:
                        table[h] = pos[t] if pos[t] >= 0 else -2
            else:
                cur = 64
                while cur < 4 * ns:
                    cur *= 2
                tmask = np.uint64(cur - 1)
                table[:cur] = -1
                for i in range(ns):
                    _insert(table, tmask, ka[i], kb[i], i)
                occ = ns
        if ns > peak:
            peak = ns
    out16[:] = 0.0
    mx = ninf
    for i in range(ns):
        if sc[i] > ninf and lw[i] > mx:
            mx = lw[i]
    if mx > ninf:
        sh = np.uint64(nd - 64)
        for i in range(ns):
            if sc[i] > ninf:
                out16[np.int64((kb[i] >> sh) & np.uint64(15))] += np.exp(lw[i] - mx)
    stats[0] = peak
    stats[1] = mx


@njit(nogil=True, cache=True)
def _mld_range(c0, c1, m0, m1, cl0, cl1, lr, v0, v1, hasv0, cdet, cv0, cv1, cw, d1z, h0, D, lo, hi, nd, leps, max_states, compact_every,
               ka, kb, lw, sc, cka, ckb, clw, csc, cs, ce, table, out16, stats):
    """Decode shots lo..hi-1 of D on the calling thread, without the GIL."""
    for s in range(lo, hi):
        D0 = np.uint64(0)
        D1 = np.uint64(0)
        for d in range(nd):
            if D[s, d]:
                if d < 64:
                    D0 |= np.uint64(1) << np.uint64(d)
                else:
                    D1 |= np.uint64(1) << np.uint64(d - 64)
        _mld_shot(c0, c1, m0, m1, cl0, cl1, lr, v0, v1, hasv0, cdet, cv0, cv1, cw, d1z, h0, D0, D1, nd, leps, max_states,
                  compact_every, ka, kb, lw, sc, cka, ckb, clw, csc, cs, ce, table, out16[s], stats[s])


# The shots are decoded by a pool of Python threads that call the kernel without the GIL. Each thread takes the next
# small batch of shots when it is free, so no thread is left waiting while another still has a pile of hard shots
# (numba's prange hands every thread one fixed slice of the shots, and the hard shots are not spread evenly).
NUM_THREADS = max(1, int(numba.config.NUMBA_NUM_THREADS))
_POOL = ThreadPoolExecutor(NUM_THREADS)
_LOCAL = threading.local()


def _scratch(max_states):
    """This thread's scratch arrays for the kernel."""
    cache = _LOCAL.__dict__.setdefault("scratch", {})
    if max_states not in cache:
        cap = 2 * max_states + 1
        tsize = 64
        while tsize < 4 * cap:
            tsize *= 2
        cache[max_states] = tuple(np.empty(cap, t) for t in (np.uint64, np.uint64, np.float64, np.float64, np.uint64, np.uint64,
                                                             np.float64, np.float64, np.float64, np.float64)) + (np.empty(tsize, np.int64),)
    return cache[max_states]


def _worker(tasks, counter):
    while True:
        i = next(counter)  # atomic: each task is taken by exactly one thread
        if i >= len(tasks):
            return
        g, D, lo, hi, leps, max_states, compact_every, out16, stats = tasks[i]
        _mld_range(g["c0"], g["c1"], g["m0"], g["m1"], g["cl0"], g["cl1"], g["lr"], g["v0"], g["v1"], g["hasv0"], g["cdet"], g["cv0"], g["cv1"],
                   g["cw"], g["d1z"], g["h0"], D, lo, hi, g["nd"], leps, max_states, compact_every, *_scratch(max_states), out16, stats)


def _run_many(sweeps, compact_every=16, priority=None):
    """sweeps: list of (g, detector_events, eps, max_states). Returns a list of (out16, stats), one per sweep. The
    batches of all sweeps share the thread pool; the result does not depend on the number of threads.

    priority (optional, one value per shot, the same shots in every sweep): an estimate of each shot's cost. The
    shots are then handed out most expensive first, in batches of about equal estimated cost, so no thread starts a
    hard shot when the others are nearly done. Every shot is decoded on its own, so this changes only which thread
    decodes it and when, never its result; the results are returned in the callers' shot order."""
    order = None
    if priority is not None:
        priority = np.asarray(priority, dtype=np.float64)
        order = np.argsort(-priority, kind="stable")
        cost = np.maximum(priority[order], 1.0)
        budget = cost.sum() / (32 * NUM_THREADS)
        bounds = [0]
        acc = 0.0
        for i in range(len(cost)):
            if i > bounds[-1] and (acc + cost[i] > budget or i - bounds[-1] >= 64):
                bounds.append(i)
                acc = 0.0
            acc += cost[i]
        bounds.append(len(cost))
    results, per_sweep = [], []
    for g, detector_events, eps, max_states in sweeps:
        D = np.ascontiguousarray(np.asarray(detector_events, dtype=np.bool_))
        if order is not None:
            D = np.ascontiguousarray(D[order])
            ranges = list(zip(bounds[:-1], bounds[1:]))
        else:
            step = max(1, min(64, len(D) // (32 * NUM_THREADS)))
            ranges = [(lo, min(lo + step, len(D))) for lo in range(0, len(D), step)]
        out16 = np.zeros((len(D), 16))
        stats = np.zeros((len(D), 2))
        results.append((out16, stats))
        per_sweep.append([(g, D, lo, hi, np.log(eps), max_states, compact_every, out16, stats) for lo, hi in ranges])
    tasks = [t for group in itertools.zip_longest(*per_sweep) for t in group if t is not None]
    counter = itertools.count()
    futures = [_POOL.submit(_worker, tasks, counter) for _ in range(min(NUM_THREADS, len(tasks)))]
    for f in futures:
        f.result()
    if order is not None:
        restored = []
        for out16, stats in results:
            o, s = np.empty_like(out16), np.empty_like(stats)
            o[order] = out16
            s[order] = stats
            restored.append((o, s))
        results = restored
    return results


def _run(g, detector_events, eps, max_states, compact_every):
    return _run_many([(g, detector_events, eps, max_states)], compact_every)[0]


def _logw(out16, stats):
    with np.errstate(divide="ignore"):
        return np.log(out16) + np.where(np.isfinite(stats[:, 1:2]), stats[:, 1:2], 0.0)


def mld_logw_many(sweeps, priority=None, return_costs=False):
    """mld_logw for several sweeps (g, detector_events, eps, max_states) at once, sharing the thread pool.
    priority: optional per-shot cost estimate, used only for scheduling (see _run_many). With return_costs, also
    returns each shot's peak state count summed over the sweeps, a cost estimate for a later, tighter stage."""
    runs = _run_many(sweeps, priority=priority)
    weights = [_logw(out16, stats) for out16, stats in runs]
    if return_costs:
        return weights, sum(stats[:, 0] for _, stats in runs)
    return weights


def mld_classp(g, detector_events, eps=1e-4, max_states=1 << 12, compact_every=16):
    """Normalised posterior over the 16 logical outcomes, shape (shots, 16), and the peak state count per shot."""
    out16, stats = _run(g, detector_events, eps, max_states, compact_every)
    tot = out16.sum(1, keepdims=True)
    return out16 / np.where(tot > 0, tot, 1), stats[:, 0]


def mld_logw(g, detector_events, eps=1e-4, max_states=1 << 12, compact_every=16):
    """Log of the unnormalised weight of each of the 16 logical outcomes, shape (shots, 16); -inf where no error
    vector of that outcome survived the pruning.

    The weights are on the same absolute scale for every sweep of the same shot (relative to the no-error vector),
    and pruning only ever removes terms from the sum. Each entry is therefore a lower bound on the exact weight, so
    the results of different sweeps (another mechanism order, another threshold) can be combined by taking the
    largest value per outcome."""
    return _logw(*_run(g, detector_events, eps, max_states, compact_every))


def marginals_logw(logw):
    """Per-bit probability of a logical flip from log outcome weights; 0.5 for shots with no surviving outcome."""
    mx = logw.max(1, keepdims=True)
    ok = np.isfinite(mx)
    w = np.where(ok, np.exp(logw - np.where(ok, mx, 0.0)), 1.0)
    return marginals(w / w.sum(1, keepdims=True))


def marginals(classp):
    """Per-bit probability of a logical flip from the 16-outcome posterior."""
    bits = (np.arange(16)[:, None] >> np.arange(4)) & 1
    return classp @ bits
