"""Canonical cached bit primitives and an exact, bounded explanation pool.

Hash collisions compare complete vectors. The max-heap preserves the original
first-index argmax tie break; each distinct error vector is weighted once.
"""
import numpy as np
from numba import njit

_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_CTZ_TABLE = np.zeros(64, np.int64)
for _i in range(64):
    _CTZ_TABLE[(((1 << _i) * int(_DEBRUIJN)) & ((1 << 64) - 1)) >> 58] = _i


@njit(cache=True)
def ctz(x):
    return _CTZ_TABLE[((x & (~x + np.uint64(1))) * _DEBRUIJN) >> np.uint64(58)]


@njit(cache=True)
def bit(x0, x1, r):
    one = np.uint64(1)
    return ((x0 >> np.uint64(r)) & one) if r < 64 else ((x1 >> np.uint64(r - 64)) & one)


@njit(cache=True)
def vector_hash(vector):
    h = np.uint64(0xCBF29CE484222325)
    for x in vector:
        h = (h ^ x) * np.uint64(0x100000001B3)
    return h


@njit(cache=True)
def create(capacity):
    size = 1
    while size < 4 * capacity:
        size *= 2
    return (np.full(size, -1, np.int64), np.zeros(1, np.int64),
            np.empty(capacity, np.int64), np.empty(capacity, np.int64))


@njit(cache=True)
def _bucket(h, mask):
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return np.int64(h & np.uint64(mask))


@njit(cache=True)
def _greater(i, j, costs):
    return costs[i] > costs[j] or (costs[i] == costs[j] and i < j)


@njit(cache=True)
def insert(vector, cost, logical, vectors, costs, logs, hashes, used, h, index):
    slots, tombs, positions, heap = index
    mask = len(slots) - 1
    if tombs[0] > len(costs):
        slots[:] = -1
        for i in range(used):
            pos = _bucket(hashes[i], mask)
            while slots[pos] >= 0:
                pos = (pos + 1) & mask
            slots[pos], positions[i] = i, pos
        tombs[0] = 0
    pos = _bucket(h, mask)
    first_tomb = -1
    while slots[pos] != -1:
        slot = slots[pos]
        if slot == -2:
            if first_tomb < 0:
                first_tomb = pos
        elif hashes[slot] == h and np.all(vectors[slot] == vector):
            return used, 0
        pos = (pos + 1) & mask
    if first_tomb >= 0:
        pos = first_tomb
    if used < len(costs):
        slot = used
        used += 1
        dropped = 0
    else:
        slot = heap[0]
        if cost >= costs[slot]:
            return used, 1
        slots[positions[slot]] = -2
        tombs[0] += 1
        dropped = 1
    if slots[pos] == -2:
        tombs[0] -= 1
    slots[pos], positions[slot] = slot, pos
    vectors[slot] = vector
    costs[slot], logs[slot], hashes[slot] = cost, logical, h
    if not dropped:
        k = used - 1
        while k:
            parent = (k - 1) // 2
            if not _greater(slot, heap[parent], costs):
                break
            heap[k] = heap[parent]
            k = parent
        heap[k] = slot
    else:
        k = 0
        while 2 * k + 1 < used:
            child = 2 * k + 1
            if child + 1 < used and _greater(heap[child + 1], heap[child], costs):
                child += 1
            if not _greater(heap[child], slot, costs):
                break
            heap[k] = heap[child]
            k = child
        heap[k] = slot
    return used, dropped


@njit(cache=True)
def put(vector, cost, logical, vectors, costs, logs, hashes, used, index):
    return insert(vector, cost, logical, vectors, costs, logs, hashes,
                  used, vector_hash(vector), index)
