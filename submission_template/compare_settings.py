"""Compare two settings of the maximum-likelihood decoder on the same fresh shots (instead of cross-validation).

The decoder has nothing to train and Stim gives unlimited labelled shots, so instead of re-using folds of data/train
this script samples fresh shots from the competition circuits and compares two settings A and B on exactly the same
shots:

- LER of each setting (only the leaderboard LER when no --min-fired filter is used).
- Paired comparison: a = bits only A gets right, b = bits only B gets right. Bits both get right or both get wrong say
  nothing about which setting is better, so only a and b matter. A two-sided sign test (McNemar) gives the p-value of
  "A and B are equally accurate"; p < 0.05 means the difference is real, not luck of the shots.
- With --reference: disagreements with a near-exact decoder (both sweep directions, very loose pruning). This measures
  each setting's own pruning error without the noise of the labels, so it shows differences of a few bits. The
  reference is slow and is cached in .cache/ for the same seed, shot count and filter.
- Runtime of each setting: median over --runs, with the settings run round-robin so a machine that heats up (or
  slows down for any other reason) affects both equally. Only decoding is timed, as in run_local.py.

A setting is a Python dict literal with any of the keys tiers, hexp and compact_every; missing keys take the values in
decoder_two_pass.py (the two-pass decoder these settings belong to; decoder.py now uses stages with a different
trigger). tiers is a list of (direction, eps, max_states, thr) passes as in decoder_two_pass.MLD_TIERS.

Run from the repository root, for example:

    # is a cheaper second pass as accurate as the current settings? (A defaults to decoder_two_pass.py)
    python submission_template/compare_settings.py --b "{'tiers': [(0, 1e-4, 1024, 0), (1, 1e-4, 2048, 0.15)]}"

    # same, on hard shots only (>= 10 detectors fired), against the near-exact reference
    python submission_template/compare_settings.py --b "{'hexp': 0.6}" --min-fired 10 --shots 20000 --reference

    # confirm a winner's LER on a large fresh set with another seed
    python submission_template/compare_settings.py --b "{...}" --shots 500000 --seed 2
"""
import argparse
import ast
import hashlib
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--a", default="{}", help="setting A as a dict literal (default: {} = the values in decoder_two_pass.py)")
parser.add_argument("--b", required=True, help="setting B as a dict literal, e.g. \"{'hexp': 0.6}\"")
parser.add_argument("--shots", type=int, default=100_000, help="shots per basis (default: 100000)")
parser.add_argument("--seed", type=int, default=1, help="sampling seed; use a new one to confirm a result (default: 1)")
parser.add_argument("--min-fired", type=int, default=0,
                    help="keep only shots with at least this many detectors fired (hard shots; default: 0 = all). "
                         "The LER is then not the leaderboard LER, but comparisons get much more signal per second.")
parser.add_argument("--reference", action="store_true", help="also count disagreements with a near-exact decoder (slow, cached)")
parser.add_argument("--runs", type=int, default=3, help="timed runs per setting; the median is reported (default: 3)")
parser.add_argument("--threads", type=int, default=None, help="numba thread count (default: all cores)")
parser.add_argument("--circuits", default=os.path.join(ROOT, "data", "train"), help="folder with X_circuit.stim and Z_circuit.stim")
args = parser.parse_args()

if args.threads:  # must be set before numba is imported
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)

import numpy as np  # noqa: E402
import stim  # noqa: E402

sys.path.insert(0, HERE)
import decoder_two_pass as decoder  # noqa: E402
from hackathon_2026 import problem_info  # noqa: E402
from mld import build_mld, marginals_logw, mld_logw  # noqa: E402

REFERENCE = dict(eps=1e-7, max_states=16384)  # near-exact: every bit agrees with tighter pruning in our tests
CACHE = os.path.join(ROOT, ".cache", "compare_settings")


def parse_setting(text):
    s = ast.literal_eval(text)
    unknown = set(s) - {"tiers", "hexp", "compact_every"}
    if unknown:
        raise SystemExit(f"unknown setting keys {sorted(unknown)}; use tiers, hexp, compact_every")
    return dict(tiers=[tuple(t) for t in s.get("tiers", decoder.MLD_TIERS)],
                hexp=float(s.get("hexp", decoder.MLD_HEXP)),
                compact_every=int(s.get("compact_every", decoder.MLD_COMPACT_EVERY)))


def sample(circuit, basis):
    """Fresh labelled shots; with --min-fired, keep sampling until enough hard shots are collected."""
    sampler = circuit.compile_detector_sampler(seed=args.seed * 2 + (basis == "Z"))
    det, obs, have = [], [], 0
    while have < args.shots:
        d, o = sampler.sample(max(100_000, args.shots), separate_observables=True)
        keep = d.sum(1) >= args.min_fired
        det.append(d[keep])
        obs.append(o[keep])
        have += int(keep.sum())
    return np.concatenate(det)[: args.shots], np.concatenate(obs)[: args.shots].astype(np.uint8)


def decode(setting, tables, D):
    """The decoder's own multi-pass logic (decoder.decode_marginals) with this setting's passes."""
    decoder.MLD_TIERS = setting["tiers"]
    decoder.MLD_COMPACT_EVERY = setting["compact_every"]
    return (decoder.decode_marginals(tables, D) > 0.5).astype(np.uint8)


def reference(tables, D, basis):
    key = hashlib.sha256(f"{basis} {args.seed} {args.shots} {args.min_fired} {REFERENCE} {args.circuits}".encode()).hexdigest()[:16]
    path = os.path.join(CACHE, f"reference_{basis}_{key}.npz")
    if os.path.exists(path):
        return np.load(path)["pred"]
    print(f"  computing the near-exact reference for {basis} (slow; cached in {path}) ...", flush=True)
    t = time.perf_counter()
    logw = np.maximum(mld_logw(tables[0], np.ascontiguousarray(D), **REFERENCE),
                      mld_logw(tables[1], np.ascontiguousarray(D[:, ::-1]), **REFERENCE))
    pred = (marginals_logw(logw) > 0.5).astype(np.uint8)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, pred=pred)
    print(f"  ... {time.perf_counter() - t:.0f} s", flush=True)
    return pred


def sign_test(a, b):
    """Two-sided exact sign test: p-value that a and b come from a fair coin (A and B equally accurate)."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    log_half_n = -n * math.log(2)
    tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n) for i in range(k + 1))
    return min(1.0, 2 * tail)


def verdict(a, b, p):
    if p >= 0.05:
        return "no significant difference"
    return f"{'A' if a > b else 'B'} is more accurate (p = {p:.2g})"


settings = {"A": parse_setting(args.a), "B": parse_setting(args.b)}
for name, s in settings.items():
    print(f"{name}: tiers={s['tiers']} hexp={s['hexp']} compact_every={s['compact_every']}")
filt = f", only shots with >= {args.min_fired} detectors fired" if args.min_fired else ""
print(f"{args.shots:,} fresh shots per basis, seed {args.seed}{filt}\n")

data = {}
for basis in "XZ":
    circuit = stim.Circuit.from_file(os.path.join(args.circuits, f"{basis}_circuit.stim"))
    m = problem_info(circuit)
    tables = {}
    for name, s in settings.items():  # each setting gets tables for its hexp (forward and backward sweep)
        if s["hexp"] not in tables:
            tables[s["hexp"]] = (build_mld(m["H"], m["L"], m["p"], hexp=s["hexp"]),
                                 build_mld(m["H"][::-1].copy(), m["L"], m["p"], hexp=s["hexp"]))
    D, O = sample(circuit, basis)
    data[basis] = dict(tables=tables, D=D, O=O)
    if args.reference:
        data[basis]["ref"] = reference(tables[settings["A"]["hexp"]], D, basis)

# decode: one untimed warm-up run (numba compile / cache load), then round-robin timed runs
pred = {}
times = {name: [] for name in settings}
for run in range(args.runs + 1):
    for name, s in settings.items():
        total = 0.0
        for basis, dd in data.items():
            t = time.perf_counter()
            p = decode(s, dd["tables"][s["hexp"]], dd["D"])
            total += time.perf_counter() - t
            if run == 0:
                pred[name, basis] = p
            elif not np.array_equal(p, pred[name, basis]):
                print(f"warning: setting {name} gave different predictions on run {run + 1} ({basis})")
        if run > 0:
            times[name].append(total)

ler_name = "LER" if not args.min_fired else "LER (hard shots only, not the leaderboard LER)"
rows = []
for basis in list(data) + ["both"]:
    bases = list(data) if basis == "both" else [basis]
    O = np.concatenate([data[x]["O"] for x in bases])
    pa = np.concatenate([pred["A", x] for x in bases])
    pb = np.concatenate([pred["B", x] for x in bases])
    a = int(np.sum((pa == O) & (pb != O)))
    b = int(np.sum((pb == O) & (pa != O)))
    p = sign_test(a, b)
    line = f"{basis:>4}: {ler_name} A {np.mean(pa != O):.5f}  B {np.mean(pb != O):.5f} | only A right {a}, only B right {b}: {verdict(a, b, p)}"
    if args.reference:
        ref = np.concatenate([data[x]["ref"] for x in bases])
        ra = int(np.sum((pa == ref) & (pb != ref)))  # only A matches the reference
        rb = int(np.sum((pb == ref) & (pa != ref)))
        line += (f" | disagree with near-exact: A {int(np.sum(pa != ref))}, B {int(np.sum(pb != ref))} bits: "
                 f"{verdict(ra, rb, sign_test(ra, rb)).replace('more accurate', 'closer to optimal')}")
    print(line)

ta, tb = statistics.median(times["A"]), statistics.median(times["B"])
print(f"\nruntime (median of {args.runs}, both bases): A {ta:.2f} s, B {tb:.2f} s (B/A = {tb / ta:.2f}); "
      f"spread A {min(times['A']):.2f}-{max(times['A']):.2f} s, B {min(times['B']):.2f}-{max(times['B']):.2f} s")
if args.reference:
    print("Disagreement with the near-exact reference is each setting's own pruning error; it has no label noise, "
          "so it is the most sensitive accuracy measure here.")
