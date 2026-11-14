"""Build the lookup tables that decoder.py loads, one per basis, from fresh seeded Stim shots.

Run from the repository root:

    uv run python submission_template/build_lut.py [--shots 50000000] [--min-count 20] [--chunk 2000000]

With the defaults this takes ~4 minutes and ~5 GB of RAM, and each table is ~1.5 MB.

For every detector pattern seen in training, each of the four logical bits is predicted by majority vote over the
shots with that pattern (ties go to 0), and the number of shots with that pattern is stored as its count; decode()
only trusts patterns with a large enough count. Patterns seen fewer than --min-count times are not stored. The tables are written to submission_template/lut/lut_<basis>.npz, together
with manifest.json, which maps a hash of each circuit's text to its table so decode() can pick the right one.

Training data is sampled straight from the circuits with fixed seeds (different from data/train's seed 42), so
data/train stays a held-out evaluation set. Stim's output for a given seed is reproducible for the same Stim version
on the same machine architecture.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = {"X": 20261113, "Z": 20261114}
N_DET, N_OBS = 70, 4
KEY_BYTES = (N_DET + 7) // 8  # 9 bytes: Stim's bit_packed layout (little-endian bits within each byte)


def circuit_hash(circuit: stim.Circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def merge(keys, n, ones):
    """Combine rows with equal keys, summing shot counts n and per-observable 1-counts."""
    uniq, inv = np.unique(keys, return_inverse=True)
    n_out = np.bincount(inv, weights=n, minlength=len(uniq)).astype(np.int64)
    ones_out = np.stack([np.bincount(inv, weights=ones[:, j], minlength=len(uniq)) for j in range(N_OBS)], 1)
    return uniq, n_out, ones_out.astype(np.int64)


def build(basis: str, circuit: stim.Circuit, shots: int, chunk: int):
    assert circuit.num_detectors == N_DET and circuit.num_observables == N_OBS
    sampler = circuit.compile_detector_sampler(seed=SEEDS[basis])
    keys = np.empty(0, dtype=f"V{KEY_BYTES}")
    n = np.empty(0, dtype=np.int64)
    ones = np.empty((0, N_OBS), dtype=np.int64)
    done = 0
    while done < shots:
        m = min(chunk, shots - done)
        det, obs = sampler.sample(m, separate_observables=True, bit_packed=True)
        k = np.ascontiguousarray(det).view(f"V{KEY_BYTES}").ravel()
        o = np.unpackbits(obs, axis=1, count=N_OBS, bitorder="little").astype(np.int64)
        keys, n, ones = merge(np.concatenate([keys, k]), np.concatenate([n, np.ones(m, np.int64)]),
                              np.concatenate([ones, o]))
        done += m
        print(f"  {basis}: {done:,} shots, {len(keys):,} distinct patterns")
    predict = (2 * ones > n[:, None]).astype(np.uint8)  # strict majority; ties -> 0
    return keys, predict, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, default=50_000_000, help="training shots per basis (default: 50M)")
    parser.add_argument("--chunk", type=int, default=2_000_000, help="shots sampled per chunk (default: 2M)")
    parser.add_argument("--min-count", type=int, default=20,
                        help="only store patterns seen at least this often (default: 20, = decoder.LUT_MIN_COUNT)")
    parser.add_argument("--circuits", default="data/train", help="folder with X_circuit.stim and Z_circuit.stim")
    parser.add_argument("--out", default=os.path.join(HERE, "lut"), help="output folder (default: submission_template/lut)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    for basis in "XZ":
        circuit = stim.Circuit.from_file(os.path.join(args.circuits, f"{basis}_circuit.stim"))
        t = time.perf_counter()
        keys, predict, n = build(basis, circuit, args.shots, args.chunk)
        keep = n >= args.min_count  # rare patterns have noisy majority votes and are never used; drop them
        keys, predict, n = keys[keep], predict[keep], n[keep]
        packed = np.packbits(predict, axis=1, bitorder="little")[:, 0]  # 4 prediction bits in one byte
        path = os.path.join(args.out, f"lut_{basis}.npz")
        np.savez(path, keys=keys.view(np.uint8).reshape(-1, KEY_BYTES), predict=packed,
                 count=np.minimum(n, np.iinfo(np.uint32).max).astype(np.uint32))
        manifest[circuit_hash(circuit)] = {"basis": basis, "file": os.path.basename(path), "seed": SEEDS[basis],
                                           "shots": args.shots, "min_count": args.min_count,
                                           "patterns": int(len(keys)),
                                           "nonzero_predictions": int((packed != 0).sum())}
        print(f"{basis}: {len(keys):,} patterns, {int((packed != 0).sum()):,} predict a flip, "
              f"{time.perf_counter() - t:.1f} s -> {path}")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
