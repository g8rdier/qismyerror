"""Belief propagation (damped normalised min-sum) with ordered-statistics decoding (OSD-0 + combination sweep), from scratch.

Decoding problem: the error model gives H (detectors x error mechanisms), L (observables x mechanisms) and the
prior probability p of each mechanism. Given detector events D, find a likely error e with H e = D (mod 2) and
predict the logical flips L e (mod 2).

1. BP: messages are passed between error nodes and detector nodes of the graph defined by H, starting from the
   priors. Detector-to-error messages are damped (mixed with the previous iteration) and the posteriors are
   averaged over the iterations, which stops min-sum oscillating on this loopy graph. By default OSD always runs
   after a fixed number of iterations; with always_osd=False the hard decision is checked against D instead.
2. OSD-0, if BP has not matched D after max_iter iterations: sort the mechanisms from most to least likely by their
   BP posterior, pick the first linearly independent set of columns of H (Gaussian elimination over GF(2)), and
   solve H e = D using only those columns. This always gives an e that explains D. The combination sweep (lam,
   lam2) also tries the most likely columns outside that set, singly and in pairs, and picks the logical outcome
   with the largest total probability over all candidates (coset=True). Shots where some logical bit is still
   uncertain (probability between refine_thr and 1 - refine_thr) are decoded again with a larger sweep.

Columns of H have 70 bits, stored as two uint64 words. Shots are decoded in parallel with numba.
"""
import numba
import numpy as np
from numba import njit, prange


def build_graph(H: np.ndarray, L: np.ndarray, p: np.ndarray):
    """Precompute everything the kernel needs from the error model."""
    H = np.asarray(H, dtype=np.uint8)
    m, n = H.shape
    assert m <= 128
    ec, ev = np.nonzero(H)  # edges sorted by check (detector), then by variable (mechanism)
    check_ptr = np.zeros(m + 1, np.int64)
    np.cumsum(np.bincount(ec, minlength=m), out=check_ptr[1:])
    order = np.argsort(ev, kind="stable")  # edge ids grouped by variable
    var_ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(ev, minlength=n), out=var_ptr[1:])
    cols = np.zeros((n, 2), np.uint64)  # column j of H as a 128-bit mask
    for i, j in zip(ec, ev):
        cols[j, i // 64] |= np.uint64(1) << np.uint64(i % 64)
    logical = (np.asarray(L, np.uint8).T * (1 << np.arange(L.shape[0]))).sum(1).astype(np.uint8)  # 4-bit mask per column
    llr0 = np.log((1 - p) / p).astype(np.float32)
    return dict(ec=ec.astype(np.int64), ev=ev.astype(np.int64), check_ptr=check_ptr, var_edges=order.astype(np.int64),
                var_ptr=var_ptr, cols=cols, logical=logical, llr0=llr0, m=m, n=n)


_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_CTZ_TABLE = np.zeros(64, np.int64)
for _i in range(64):
    _CTZ_TABLE[((np.uint64(1) << np.uint64(_i)) * _DEBRUIJN) >> np.uint64(58)] = _i


@njit(cache=True)
def _ctz(x):
    """Index of the lowest set bit of a non-zero uint64 (de Bruijn multiplication)."""
    return _CTZ_TABLE[((x & (~x + np.uint64(1))) * _DEBRUIJN) >> np.uint64(58)]


@njit(cache=True)
def _bit(x0, x1, r):
    one = np.uint64(1)
    return ((x0 >> np.uint64(r)) & one) if r < 64 else ((x1 >> np.uint64(r - 64)) & one)


@njit(cache=True)
def _osd(post, cost, cols, logical, D0, D1, m, lam, lam2, coset, cp, temp=1.0):
    """Ordered-statistics decoding.

    Columns of H are taken in order of BP posterior (most likely first); the first m independent ones form the
    information set S. The OSD-0 solution uses only S. With lam > 0, the combination sweep also tries switching on
    each of the lam most likely columns outside S (and every pair among the first lam2 of them), re-solving S each
    time. Every candidate explains D exactly; its cost is sum(cost[j]) over its errors (cost = -log-likelihood
    ratio). Returns the logical mask of the cheapest candidate, or, with coset=True, the logical mask whose
    candidates have the largest total probability.
    """
    b0 = np.empty(m, np.uint64)  # basis, kept fully reduced: each vector has exactly one pivot bit set
    b1 = np.empty(m, np.uint64)
    c0 = np.empty(m, np.uint64)  # which selected columns make up each basis vector (bitmask over <= 128 slots)
    c1 = np.empty(m, np.uint64)
    sel = np.empty(m, np.int64)
    kidx = np.empty(128, np.int64)  # basis vector whose pivot is this detector bit
    nl = max(lam, 1)
    npj = np.empty(nl, np.int64)  # most likely columns outside S, and their expansion in terms of S
    nk0 = np.empty(nl, np.uint64)
    nk1 = np.empty(nl, np.uint64)
    # Only the first ~m + lam columns in likelihood order are ever used, so sort just the most likely K of them;
    # in the rare case that is not enough, redo the elimination with a full sort.
    n = post.shape[0]
    K = min(n, m + lam + 128)
    for attempt in range(2):
        if K < n and attempt == 0:
            thr = np.partition(post.copy(), K - 1)[K - 1]
            cand = np.nonzero(post <= thr)[0]
            order = cand[np.argsort(post[cand])]
        else:
            order = np.argsort(post)
        pm0 = np.uint64(0)
        pm1 = np.uint64(0)
        nn = 0
        rank = 0
        for t in range(order.shape[0]):
            if rank == m and nn >= lam:
                break
            j = order[t]
            v0 = cols[j, 0]
            v1 = cols[j, 1]
            k0 = np.uint64(0)
            k1 = np.uint64(0)
            for w in range(2):  # only the pivot bits this column touches need work
                x = (v0 & pm0) if w == 0 else (v1 & pm1)
                while x:
                    k = kidx[_ctz(x) + 64 * w]
                    x &= x - np.uint64(1)
                    v0 ^= b0[k]
                    v1 ^= b1[k]
                    k0 ^= c0[k]
                    k1 ^= c1[k]
            if v0 == 0 and v1 == 0:
                if nn < lam:  # column j = XOR of the selected columns in k
                    npj[nn] = j
                    nk0[nn] = k0
                    nk1[nn] = k1
                    nn += 1
                continue
            # new pivot: lowest set bit of the reduced column (never an existing pivot)
            if v0:
                pk = _ctz(v0)
                pb0 = np.uint64(1) << np.uint64(pk)
                pb1 = np.uint64(0)
            else:
                pk = 64 + _ctz(v1)
                pb0 = np.uint64(0)
                pb1 = np.uint64(1) << np.uint64(pk - 64)
            if rank < 64:
                k0 ^= np.uint64(1) << np.uint64(rank)
            else:
                k1 ^= np.uint64(1) << np.uint64(rank - 64)
            for k in range(rank):  # clear the new pivot from the other basis vectors
                if (b0[k] & pb0) | (b1[k] & pb1):
                    b0[k] ^= v0
                    b1[k] ^= v1
                    c0[k] ^= k0
                    c1[k] ^= k1
            b0[rank] = v0
            b1[rank] = v1
            c0[rank] = k0
            c1[rank] = k1
            sel[rank] = j
            kidx[pk] = rank
            pm0 |= pb0
            pm1 |= pb1
            rank += 1
        if rank == m and nn >= lam:
            break
    # solve: with a fully reduced basis, D is the XOR of the basis vectors at its pivot bits
    kd0 = np.uint64(0)
    kd1 = np.uint64(0)
    for w in range(2):
        x = (D0 & pm0) if w == 0 else (D1 & pm1)
        while x:
            k = kidx[_ctz(x) + 64 * w]
            x &= x - np.uint64(1)
            kd0 ^= c0[k]
            kd1 ^= c1[k]
    if lam == 0 and coset == 0:
        out = np.uint8(0)
        for r in range(rank):
            if _bit(kd0, kd1, r):
                out ^= logical[sel[r]]
        cp[out] = 1.0
        return out
    # base candidate; sgnc[r] = how the cost changes when selected column r is toggled
    base_cost = 0.0
    base_log = np.uint8(0)
    sgnc = np.empty(rank, np.float64)
    for r in range(rank):
        cr = cost[sel[r]]
        if _bit(kd0, kd1, r):
            base_cost += cr
            base_log ^= logical[sel[r]]
            sgnc[r] = -cr
        else:
            sgnc[r] = cr
    best_cost = base_cost
    best_log = base_log
    prob = np.zeros(16, np.float64)
    prob[base_log] += 1.0
    # singles: flip one outside column a; S part changes by nk[a]
    dcost = np.empty(nn, np.float64)
    dlog = np.empty(nn, np.uint8)
    for a in range(nn):
        dc = cost[npj[a]]
        dl = logical[npj[a]]
        for w in range(2):
            x = nk0[a] if w == 0 else nk1[a]
            while x:
                r = _ctz(x) + 64 * w
                x &= x - np.uint64(1)
                dc += sgnc[r]
                dl ^= logical[sel[r]]
        dcost[a] = dc
        dlog[a] = dl
        c = base_cost + dc
        lg = base_log ^ dl
        prob[lg] += np.exp(temp * (base_cost - c))
        if c < best_cost:
            best_cost = c
            best_log = lg
    # pairs among the first lam2: S part changes by nk[a] ^ nk[b]; bits in both cancel, so undo them
    n2 = min(lam2, nn)
    for a in range(n2):
        for bb in range(a + 1, n2):
            c = base_cost + dcost[a] + dcost[bb]
            for w in range(2):
                x = (nk0[a] & nk0[bb]) if w == 0 else (nk1[a] & nk1[bb])
                while x:
                    r = _ctz(x) + 64 * w
                    x &= x - np.uint64(1)
                    c -= 2.0 * sgnc[r]
            lg = base_log ^ dlog[a] ^ dlog[bb]
            prob[lg] += np.exp(temp * (base_cost - c))
            if c < best_cost:
                best_cost = c
                best_log = lg
    ptot = prob.sum()
    for cls in range(16):  # probability of each 4-bit logical outcome over the candidates (cp is written for callers)
        cp[cls] = prob[cls] / ptot
    if coset == 1:  # most probable 4-bit outcome
        return np.uint8(np.argmax(prob))
    if coset == 2:  # each bit on its own: flipped if the candidates flipping it carry most of the probability
        tot = prob.sum()
        out = np.uint8(0)
        for bit in range(4):
            pb = 0.0
            for cls in range(16):
                if (cls >> bit) & 1:
                    pb += prob[cls]
            if 2 * pb > tot:
                out |= np.uint8(1 << bit)
        return out
    return best_log


@njit(parallel=True, cache=True)
def decode_batch(D, ec, ev, check_ptr, cols, logical, llr0, max_iter, alpha, always_osd, damping, avg_post, nchunks,
                 lam, lam2, coset, stats, classp, temp):
    """D: (shots, m) uint8. Returns (shots,) uint8 logical masks. stats[s] = BP iterations used, or -1 if OSD ran.

    Flooding min-sum where the error->detector message is never stored: it is post[j] - c2v[e], the posterior of
    the previous iteration minus this edge's own incoming message. Only c2v (one float per edge) and the posteriors
    are kept, so the working set stays in cache and no variable-major indirection is needed.
    """
    shots, m = D.shape
    n = llr0.shape[0]
    E = ec.shape[0]
    out = np.zeros(shots, np.uint8)
    a = np.float32(alpha)
    dmp = np.float32(damping)
    big = np.float32(1e30)
    maxdeg = 0
    for i in range(m):
        maxdeg = max(maxdeg, check_ptr[i + 1] - check_ptr[i])
    per = (shots + nchunks - 1) // nchunks
    for ch in prange(nchunks):
        c2v = np.empty(E, np.float32)
        post = np.empty(n, np.float32)
        acc = np.empty(n, np.float32)  # sum of posteriors over iterations (used when avg_post)
        hard = np.empty(n, np.uint8)
        xs = np.empty(maxdeg, np.float32)
        for s in range(ch * per, min(shots, (ch + 1) * per)):
            d = D[s]
            for e in range(E):
                c2v[e] = 0.0
            for j in range(n):
                post[j] = llr0[j]
                acc[j] = 0.0
            converged = False
            it = 0
            while it < max_iter:
                it += 1
                # detector -> error: damped, normalised min-sum
                for i in range(m):
                    lo = check_ptr[i]
                    deg = check_ptr[i + 1] - lo
                    neg = d[i]  # parity of negative incoming messages, starting from the detector value
                    min1 = big
                    min2 = big
                    arg = -1
                    for k in range(deg):  # branch-light: gather each message once
                        x = post[ev[lo + k]] - c2v[lo + k]
                        xs[k] = x
                        neg ^= x < 0
                        ax = abs(x)
                        min2 = min(min2, max(min1, ax))
                        if ax < min1:
                            arg = k
                        min1 = min(min1, ax)
                    scale = (1 - dmp) * a * (np.float32(-1.0) if neg else np.float32(1.0))
                    for k in range(deg):
                        mag = min2 if k == arg else min1
                        sg = -scale if xs[k] < 0 else scale
                        c2v[lo + k] = dmp * c2v[lo + k] + sg * mag
                # posteriors = prior + all incoming messages
                for j in range(n):
                    post[j] = llr0[j]
                for e in range(E):
                    post[ev[e]] += c2v[e]
                for j in range(n):
                    acc[j] += post[j]
                if always_osd:
                    continue  # OSD runs regardless, so skip the convergence check
                # does the hard decision explain the detectors?
                for j in range(n):
                    hard[j] = 1 if post[j] < 0 else 0
                ok = True
                for i in range(m):
                    par = 0
                    for e in range(check_ptr[i], check_ptr[i + 1]):
                        par ^= hard[ev[e]]
                    if par != d[i]:
                        ok = False
                        break
                if ok:
                    converged = True
                    break
            if converged:
                o = np.uint8(0)
                for j in range(n):
                    if hard[j]:
                        o ^= logical[j]
                out[s] = o
                stats[s] = it
            else:
                D0 = np.uint64(0)
                D1 = np.uint64(0)
                for i in range(m):
                    if d[i]:
                        if i < 64:
                            D0 |= np.uint64(1) << np.uint64(i)
                        else:
                            D1 |= np.uint64(1) << np.uint64(i - 64)
                out[s] = _osd(acc if avg_post else post, llr0, cols, logical, D0, D1, m, lam, lam2, coset, classp[s], temp)
                stats[s] = -1
    return out


def decode_bp_osd(graph, detector_events, max_iter=3, alpha=0.8, always_osd=True, damping=0.5, avg_post=True,
                  lam=400, lam2=100, coset=2, temp=1.0, refine_thr=0.3, refine_lam=1000, refine_lam2=300,
                  return_stats=False, return_classp=False):
    D = np.ascontiguousarray(detector_events, dtype=np.uint8)
    stats = np.zeros(len(D), np.int64)
    classp = np.zeros((len(D), 16), np.float32)  # P(4-bit logical outcome) from the OSD candidates
    nchunks = max(1, min(len(D), 8 * numba.get_num_threads()))
    coset = int(coset)
    masks = decode_batch(D, graph["ec"], graph["ev"], graph["check_ptr"], graph["cols"], graph["logical"],
                         graph["llr0"], max_iter, alpha, always_osd, damping, avg_post, nchunks,
                         lam, lam2, coset, stats, classp, temp)
    if refine_thr > 0:
        # second stage: shots where some logical bit is still uncertain get a larger combination sweep
        bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)
        marg = classp @ bits
        unc = np.flatnonzero((np.abs(marg - 0.5) < 0.5 - refine_thr).any(1))
        if len(unc):
            stats2 = np.zeros(len(unc), np.int64)
            classp2 = np.zeros((len(unc), 16), np.float32)
            nch2 = max(1, min(len(unc), 8 * numba.get_num_threads()))
            masks[unc] = decode_batch(D[unc], graph["ec"], graph["ev"], graph["check_ptr"], graph["cols"],
                                      graph["logical"], graph["llr0"], max_iter, alpha, always_osd, damping, avg_post,
                                      nch2, refine_lam, refine_lam2, coset, stats2, classp2, temp)
            classp[unc] = classp2
    out = np.unpackbits(masks[:, None], axis=1, count=4, bitorder="little")
    if return_classp:
        return out, classp
    return (out, stats) if return_stats else out
