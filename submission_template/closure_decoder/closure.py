"""Physical pair index and one round of syndrome-preserving pair closure.

If H_i XOR H_j == H_k XOR H_l, the two pairs have the same detector effect.
Expand cheap candidates per logical class and retain original prior costs.
The pair index is built once at import/setup, without training samples.
"""
import hashlib
import numpy as np
from numba import njit

from closure_decoder.candidates import ctz, insert, vector_hash


@njit(cache=True)
def _slot(lo, hi, mask):
    x = lo ^ (hi * np.uint64(0x9E3779B97F4A7C15))
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return np.int64(x & np.uint64(mask))


@njit(cache=True)
def find(lo, hi, cols, heads, ii, jj):
    mask = len(heads) - 1
    slot = _slot(lo, hi, mask)
    while heads[slot] >= 0:
        k = heads[slot]
        a, b = ii[k], jj[k]
        if (cols[a, 0] ^ cols[b, 0]) == lo and (cols[a, 1] ^ cols[b, 1]) == hi:
            return k
        slot = (slot + 1) & mask
    return -1


@njit(cache=True)
def build_index(cols):
    n = len(cols)
    count = n * (n - 1) // 2
    size = 1
    while size < 2 * count:
        size *= 2
    heads = np.full(size, -1, np.int32)
    nxt = np.full(count, -1, np.int32)
    ii = np.empty(count, np.int32)
    jj = np.empty(count, np.int32)
    mask, k = size - 1, 0
    for a in range(n):
        for b in range(a + 1, n):
            lo, hi = cols[a, 0] ^ cols[b, 0], cols[a, 1] ^ cols[b, 1]
            slot = _slot(lo, hi, mask)
            while heads[slot] >= 0:
                prev = heads[slot]
                i, j = ii[prev], jj[prev]
                if ((cols[i, 0] ^ cols[j, 0]) == lo
                        and (cols[i, 1] ^ cols[j, 1]) == hi):
                    break
                slot = (slot + 1) & mask
            ii[k], jj[k] = a, b
            nxt[k] = heads[slot]
            heads[slot] = k
            k += 1
    return heads, nxt, ii, jj


_INDICES = {}


def get_index(cols):
    key = hashlib.sha256(cols.tobytes()).hexdigest()
    if key not in _INDICES:
        _INDICES[key] = build_index(cols)
    return _INDICES[key]


@njit(cache=True)
def expand(cols, logical, prior, heads, nxt, ii, jj,
           vectors, costs, logs, hashes, used, gap, index, seed_count):
    order = np.argsort(costs[:used])
    count = 0
    maximum = min(used, seed_count * 16)
    seeds = np.empty(maximum, np.int64)
    counts = np.zeros(16, np.int64)
    for i in order:
        if count == maximum:
            break
        if counts[logs[i]] < seed_count:
            seeds[count] = i
            counts[logs[i]] += 1
            count += 1
    # Pool replacements below must not mutate our seed explanations.
    vv, cc, ll = vectors[seeds[:count]].copy(), costs[seeds[:count]].copy(), logs[seeds[:count]].copy()
    best = costs[:used].min()
    drops = 0
    for s in range(count):
        present = np.empty(len(prior), np.int64)
        n = 0
        for word in range(vv.shape[1]):
            x = vv[s, word]
            while x:
                r = ctz(x) + 64 * word
                x &= x - np.uint64(1)
                present[n] = r
                n += 1
        for a in range(n):
            i = present[a]
            for b in range(a + 1, n):
                j = present[b]
                lo, hi = cols[i, 0] ^ cols[j, 0], cols[i, 1] ^ cols[j, 1]
                k = find(lo, hi, cols, heads, ii, jj)
                while k >= 0:
                    u, v = np.int64(ii[k]), np.int64(jj[k])
                    if i == u and j == v:
                        k = nxt[k]
                        continue
                    cost = cc[s] - prior[i] - prior[j]
                    on_u = ((vv[s, u // 64] >> np.uint64(u % 64)) & np.uint64(1)) != 0
                    on_v = ((vv[s, v // 64] >> np.uint64(v % 64)) & np.uint64(1)) != 0
                    if u == i or u == j:
                        on_u = not on_u
                    if v == i or v == j:
                        on_v = not on_v
                    cost += -prior[u] if on_u else prior[u]
                    cost += -prior[v] if on_v else prior[v]
                    best = min(best, cost)
                    if cost <= best + gap:
                        vec = vv[s].copy()
                        lg = ll[s]
                        for r in (i, j, u, v):
                            vec[r // 64] ^= np.uint64(1) << np.uint64(r % 64)
                            lg ^= logical[r]
                        used, drop = insert(vec, cost, lg, vectors, costs, logs, hashes,
                                            used, vector_hash(vec), index)
                        drops += drop
                    k = nxt[k]
    return used, drops
