"""Damped min-sum rankings, GF(2) OSD candidates and logical marginal voting.

Only the released path is included: shared BP checkpoints, a deduplicated
singles/pairs union, then one physical-pair closure pass. Every candidate is
scored with the original mechanism priors, not BP's approximate posteriors.
"""
import numpy as np
from numba import njit, prange

from closure_decoder.candidates import bit, ctz, create, put
from closure_decoder.closure import expand, get_index


def make_graph(H, L, p):
    H, L, p = np.asarray(H, np.uint8), np.asarray(L, np.uint8), np.asarray(p, float)
    m, n = H.shape
    if m > 128 or L.shape != (4, n) or p.shape != (n,) or not ((p > 0) & (p < 1)).all():
        raise ValueError("invalid physical model dimensions/probabilities")
    ec, ev = np.nonzero(H)
    ptr = np.zeros(m + 1, np.int64)
    np.cumsum(np.bincount(ec, minlength=m), out=ptr[1:])
    cols = np.zeros((n, 2), np.uint64)
    for i, j in zip(ec, ev):
        cols[j, i // 64] |= np.uint64(1) << np.uint64(i % 64)
    logical = (L.T * (1 << np.arange(4))).sum(1).astype(np.uint8)
    return {"ev": ev.astype(np.int64), "ptr": ptr, "cols": cols, "logical": logical,
            "prior": np.log((1 - p) / p).astype(np.float32), "m": m}


@njit(cache=True)
def posteriors(D, ev, check_ptr, prior, schedules):
    E, n, count = len(ev), len(prior), len(schedules)
    output = np.empty((count, n), np.float32)
    done = np.zeros(count, np.uint8)
    big = np.float32(1e30)
    for group in range(count):
        if done[group]:
            continue
        alpha, damping = schedules[group, 1], schedules[group, 2]
        stop = 0
        for j in range(count):
            if schedules[j, 1] == alpha and schedules[j, 2] == damping:
                stop = max(stop, int(schedules[j, 0]))
                if schedules[j, 0] == 0:
                    output[j] = 0
                    done[j] = 1
        c2v = np.zeros(E, np.float32)
        post = prior.copy()
        acc = np.zeros(n, np.float32)
        xs = np.empty(E, np.float32)
        a, damp = np.float32(alpha), np.float32(damping)
        for it in range(1, stop + 1):
            for i in range(len(D)):
                lo, hi = check_ptr[i], check_ptr[i + 1]
                neg = D[i]
                min1, min2, arg = big, big, -1
                for e in range(lo, hi):
                    x = post[ev[e]] - c2v[e]
                    xs[e] = x
                    neg ^= x < 0
                    ax = abs(x)
                    min2 = min(min2, max(min1, ax))
                    if ax < min1:
                        arg = e
                    min1 = min(min1, ax)
                scale = (1 - damp) * a * (np.float32(-1) if neg else np.float32(1))
                for e in range(lo, hi):
                    mag = min2 if e == arg else min1
                    sg = -scale if xs[e] < 0 else scale
                    c2v[e] = damp * c2v[e] + sg * mag
            post[:] = prior
            for e in range(E):
                post[ev[e]] += c2v[e]
            acc += post
            for j in range(count):
                if (int(schedules[j, 0]) == it and schedules[j, 1] == alpha
                        and schedules[j, 2] == damping):
                    output[j] = acc
                    done[j] = 1
    return output


@njit(cache=True)
def basis(post, cols, D0, D1, m, lam):
    n = len(post)
    b0, b1 = np.empty(m, np.uint64), np.empty(m, np.uint64)
    c0, c1 = np.empty(m, np.uint64), np.empty(m, np.uint64)
    sel = np.empty(m, np.int64)
    kidx = np.empty(128, np.int64)
    outside = np.empty(lam, np.int64)
    nk0, nk1 = np.empty(lam, np.uint64), np.empty(lam, np.uint64)
    K = min(n, m + lam + 128)
    for attempt in range(2):
        if K < n and attempt == 0:
            threshold = np.partition(post.copy(), K - 1)[K - 1]
            cand = np.nonzero(post <= threshold)[0]
            order = cand[np.argsort(post[cand])]
        else:
            order = np.argsort(post)
        rank, nn = 0, 0
        pm0, pm1 = np.uint64(0), np.uint64(0)
        for t in range(len(order)):
            if rank == m and nn >= lam:
                break
            j = order[t]
            v0, v1 = cols[j, 0], cols[j, 1]
            k0, k1 = np.uint64(0), np.uint64(0)
            for w in range(2):
                x = (v0 & pm0) if w == 0 else (v1 & pm1)
                while x:
                    k = kidx[ctz(x) + 64 * w]
                    x &= x - np.uint64(1)
                    v0 ^= b0[k]
                    v1 ^= b1[k]
                    k0 ^= c0[k]
                    k1 ^= c1[k]
            if v0 == 0 and v1 == 0:
                if nn < lam:
                    outside[nn], nk0[nn], nk1[nn] = j, k0, k1
                    nn += 1
                continue
            if v0:
                pk = ctz(v0)
                pb0, pb1 = np.uint64(1) << np.uint64(pk), np.uint64(0)
            else:
                pk = 64 + ctz(v1)
                pb0, pb1 = np.uint64(0), np.uint64(1) << np.uint64(pk - 64)
            if rank < 64:
                k0 ^= np.uint64(1) << np.uint64(rank)
            else:
                k1 ^= np.uint64(1) << np.uint64(rank - 64)
            for k in range(rank):
                if (b0[k] & pb0) | (b1[k] & pb1):
                    b0[k] ^= v0
                    b1[k] ^= v1
                    c0[k] ^= k0
                    c1[k] ^= k1
            b0[rank], b1[rank] = v0, v1
            c0[rank], c1[rank] = k0, k1
            sel[rank], kidx[pk] = j, rank
            pm0 |= pb0
            pm1 |= pb1
            rank += 1
        if rank == m and nn >= lam:
            break
    kd0, kd1 = np.uint64(0), np.uint64(0)
    for w in range(2):
        x = (D0 & pm0) if w == 0 else (D1 & pm1)
        while x:
            k = kidx[ctz(x) + 64 * w]
            x &= x - np.uint64(1)
            kd0 ^= c0[k]
            kd1 ^= c1[k]
    return sel[:rank], outside[:nn], nk0[:nn], nk1[:nn], kd0, kd1


@njit(cache=True)
def make_vector(sel, z0, z1, j, k, words):
    vector = np.zeros(words, np.uint64)
    for w in range(2):
        x = z0 if w == 0 else z1
        while x:
            r = ctz(x) + 64 * w
            x &= x - np.uint64(1)
            col = sel[r]
            vector[col // 64] ^= np.uint64(1) << np.uint64(col % 64)
    if j >= 0:
        vector[j // 64] ^= np.uint64(1) << np.uint64(j % 64)
    if k >= 0:
        vector[k // 64] ^= np.uint64(1) << np.uint64(k % 64)
    return vector


@njit(cache=True)
def collect(post, prior, cols, logical, D0, D1, m, lam, lam2,
            vectors, costs, logs, hashes, used, gap, index):
    sel, outside, k0, k1, z0, z1 = basis(post, cols, D0, D1, m, lam)
    rank, nn = len(sel), len(outside)
    sgn = np.empty(rank, np.float64)
    basecost, baselog = 0.0, np.uint8(0)
    for r in range(rank):
        c = prior[sel[r]]
        if bit(z0, z1, r):
            basecost += c
            baselog ^= logical[sel[r]]
            sgn[r] = -c
        else:
            sgn[r] = c
    best = min(basecost, costs[:used].min()) if used else basecost
    words = vectors.shape[1]
    if basecost <= best + gap:
        vec = make_vector(sel, z0, z1, -1, -1, words)
        used, _ = put(vec, basecost, baselog, vectors, costs, logs, hashes, used, index)
    dc, dl = np.empty(nn, np.float64), np.empty(nn, np.uint8)
    for a in range(nn):
        ca, la = float(prior[outside[a]]), logical[outside[a]]
        for w in range(2):
            x = k0[a] if w == 0 else k1[a]
            while x:
                r = ctz(x) + 64 * w
                x &= x - np.uint64(1)
                ca += sgn[r]
                la ^= logical[sel[r]]
        dc[a], dl[a] = ca, la
        cost = basecost + ca
        best = min(best, cost)
        if cost <= best + gap:
            vec = make_vector(sel, z0 ^ k0[a], z1 ^ k1[a], outside[a], -1, words)
            used, _ = put(vec, cost, baselog ^ la, vectors, costs, logs, hashes, used, index)
    n2 = min(lam2, nn)
    for a in range(n2):
        for b in range(a + 1, n2):
            cost = basecost + dc[a] + dc[b]
            for w in range(2):
                x = (k0[a] & k0[b]) if w == 0 else (k1[a] & k1[b])
                while x:
                    r = ctz(x) + 64 * w
                    x &= x - np.uint64(1)
                    cost -= 2.0 * sgn[r]
            best = min(best, cost)
            if cost <= best + gap:
                vec = make_vector(sel, z0 ^ k0[a] ^ k0[b], z1 ^ k1[a] ^ k1[b],
                                  outside[a], outside[b], words)
                used, _ = put(vec, cost, baselog ^ dl[a] ^ dl[b],
                              vectors, costs, logs, hashes, used, index)
    return used


@njit(parallel=True, cache=True)
def _batch(D, ev, ptr, cols, logical, prior, schedules, lam, lam2, capacity, gap,
           heads, nxt, ii, jj, closure_seeds):
    cp = np.zeros((len(D), 16), np.float32)
    for s in prange(len(D)):
        D0, D1 = np.uint64(0), np.uint64(0)
        for r in range(D.shape[1]):
            if D[s, r]:
                if r < 64:
                    D0 |= np.uint64(1) << np.uint64(r)
                else:
                    D1 |= np.uint64(1) << np.uint64(r - 64)
        vectors = np.empty((capacity, (len(prior) + 63) // 64), np.uint64)
        costs = np.empty(capacity, np.float64)
        logs = np.empty(capacity, np.uint8)
        hashes = np.empty(capacity, np.uint64)
        index = create(capacity)
        used = 0
        snapshots = posteriors(D[s], ev, ptr, prior, schedules)
        for i in range(len(schedules)):
            used = collect(snapshots[i], prior, cols, logical, D0, D1, D.shape[1],
                           lam, lam2, vectors, costs, logs, hashes, used, gap, index)
        used, _ = expand(cols, logical, prior, heads, nxt, ii, jj,
                          vectors, costs, logs, hashes, used, gap, index, closure_seeds)
        best = costs[:used].min()
        mass = np.zeros(16, np.float64)
        for i in range(used):
            if costs[i] <= best + gap:
                mass[logs[i]] += np.exp(best - costs[i])
        cp[s] = mass / mass.sum()
    return cp


def solve(graph, detector_events, parameters, return_probabilities=False):
    D = np.ascontiguousarray(detector_events, dtype=np.uint8)
    index = get_index(graph["cols"])
    cp = _batch(D, graph["ev"], graph["ptr"], graph["cols"], graph["logical"],
                graph["prior"], np.asarray(parameters["schedules"], dtype=np.float64),
                parameters["lam"], parameters["lam2"], parameters["capacity"],
                parameters["gap"], *index, parameters["closure_seeds"])
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)
    out = (cp @ bits > 0.5).astype(np.uint8)
    return (out, cp) if return_probabilities else out
