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
import numpy as np
import numba
from numba import njit, prange

# Self-contained unsigned bit index; Python-int construction avoids overflow warnings.
_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_CTZ_TABLE = np.zeros(64, np.int64)
for _i in range(64):
    _CTZ_TABLE[(((1 << _i) * int(_DEBRUIJN)) & ((1 << 64) - 1)) >> 58] = _i


@njit(cache=True)
def _ctz(x):
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
        if t < 0:
            return -1
        if ka[t] == a and kb[t] == b:
            return t
        h = np.int64((np.uint64(h) + np.uint64(1)) & tmask)


@njit(cache=True, inline="always")
def _logadd(x, y):
    if x < y:
        x, y = y, x
    if y == -np.inf:
        return x
    return x + np.log1p(np.exp(y - x))


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
                if 2 * ns > cur:  # keep the load factor below 1/2
                    cur *= 2
                    tmask = np.uint64(cur - 1)
                    table[:cur] = -1
                    for q in range(ns):
                        _insert(table, tmask, ka[q], kb[q], q)
                else:
                    _insert(table, tmask, a, b, t)
            if sc[t] > nbest:
                nbest = sc[t]
        best = nbest
        since += 1
        if ns > max_states or since >= compact_every or closes:
            since = 0
            thr = best + leps
            m = 0
            for i in range(ns):
                if sc[i] >= thr:
                    ka[m] = ka[i]
                    kb[m] = kb[i]
                    lw[m] = lw[i]
                    sc[m] = sc[i]
                    m += 1
            ns = m
            if ns == 0:
                break
            if ns > max_states:
                idx = np.argsort(-sc[:ns])[:max_states]
                cka[:max_states] = ka[idx]
                ckb[:max_states] = kb[idx]
                clw[:max_states] = lw[idx]
                csc[:max_states] = sc[idx]
                ka[:max_states] = cka[:max_states]
                kb[:max_states] = ckb[:max_states]
                lw[:max_states] = clw[:max_states]
                sc[:max_states] = csc[:max_states]
                ns = max_states
            table[:cur] = -1
            cur = 64
            while cur < 4 * ns:
                cur *= 2
            tmask = np.uint64(cur - 1)
            table[:cur] = -1
            for i in range(ns):
                _insert(table, tmask, ka[i], kb[i], i)
        if ns > peak:
            peak = ns
    out16[:] = 0.0
    mx = ninf
    detector_hi = (np.uint64(1) << np.uint64(nd - 64)) - np.uint64(1)
    for i in range(ns):
        if ka[i] == 0 and (kb[i] & detector_hi) == 0 and sc[i] > ninf and lw[i] > mx:
            mx = lw[i]
    if mx > ninf:
        sh = np.uint64(nd - 64)
        for i in range(ns):
            if ka[i] == 0 and (kb[i] & detector_hi) == 0 and sc[i] > ninf:
                out16[np.int64((kb[i] >> sh) & np.uint64(15))] += np.exp(lw[i] - mx)
    stats[0] = peak
    stats[1] = mx


@njit(parallel=True, cache=True)
def _mld_batch(c0, c1, m0, m1, cl0, cl1, lr, v0, v1, hasv0, cdet, cv0, cv1, cw, d1z, h0, D, nd, leps, max_states, compact_every, nchunks,
               out16, stats):
    n = D.shape[0]
    cap = 2 * max_states + 1
    tsize = 64
    while tsize < 4 * cap:
        tsize *= 2
    for ch in prange(nchunks):
        ka = np.empty(cap, np.uint64)
        kb = np.empty(cap, np.uint64)
        lw = np.empty(cap)
        sc = np.empty(cap)
        cka = np.empty(cap, np.uint64)
        ckb = np.empty(cap, np.uint64)
        clw = np.empty(cap)
        csc = np.empty(cap)
        cs = np.empty(cap)
        ce = np.empty(cap)
        table = np.empty(tsize, np.int64)
        for s in range(ch * n // nchunks, (ch + 1) * n // nchunks):
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


def _run(g, detector_events, eps, max_states, compact_every):
    D = np.ascontiguousarray(np.asarray(detector_events, dtype=np.bool_))
    out16 = np.zeros((len(D), 16))
    stats = np.zeros((len(D), 2))
    nchunks = max(1, min(len(D), 16 * numba.get_num_threads()))
    _mld_batch(g["c0"], g["c1"], g["m0"], g["m1"], g["cl0"], g["cl1"], g["lr"], g["v0"], g["v1"], g["hasv0"], g["cdet"], g["cv0"], g["cv1"], g["cw"], g["d1z"],
               g["h0"], D, g["nd"], np.log(eps), max_states, compact_every, nchunks, out16, stats)
    return out16, stats


def mld_logw(g, detector_events, eps=1e-4, max_states=1 << 12, compact_every=16):
    """Log of the unnormalised weight of each of the 16 logical outcomes, shape (shots, 16); -inf where no error
    vector of that outcome survived the pruning.

    The weights are on the same absolute scale for every sweep of the same shot (relative to the no-error vector),
    and pruning only ever removes terms from the sum. Each entry is therefore a lower bound on the exact weight, so
    the results of different sweeps (another mechanism order, another threshold) can be combined by taking the
    largest value per outcome."""
    out16, stats = _run(g, detector_events, eps, max_states, compact_every)
    with np.errstate(divide="ignore"):
        return np.log(out16) + np.where(np.isfinite(stats[:, 1:2]), stats[:, 1:2], 0.0)


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
