"""Two-pass near-maximum-likelihood decoder (the faster, less accurate alternative to decoder.py; this was decoder.py
until the three-stage decoder replaced it, and compare_settings.py tunes its settings).

The top 10 teams hand in this folder: decoder.py, requirements.txt and any files decode() loads, together with
the code that created them. We call decode() once per basis on that basis's 100,000 shots in data/kaggle_private and time the two
calls (see "Handing in your code" in the README). Importing the decoder, and anything it builds when imported, is
not timed.

Every shot is decoded by the pruned dynamic programme in mld.py, which computes the weight of each of the 16 logical
outcomes and predicts each logical bit by its marginal (the optimal decision for the per-bit LER). It runs in
passes: the first decodes every shot, and each later pass re-decodes the shots where some logical bit is still
uncertain.

Pruning only removes terms from an outcome's weight, so every pass gives a lower bound on it, and passes are
combined by keeping the largest bound per outcome. Passes alternate between sweeping the error mechanisms forward
and backward in time: the two directions prune different error vectors, so their combination is closer to the exact
weights than one sweep with a tighter threshold that costs the same.

At import we rebuild the competition's two circuits (5 rounds, SI1000 p = 0.002), precompute the decoder tables for
each and compile the numba kernel, so decode() only has to look the tables up by a hash of the circuit text. A
circuit that is not one of these is handled too, by building its tables inside decode().
"""
import hashlib

import numpy as np
import stim
from hackathon_2026 import apply_circuit_noise, create_circuit, problem_info

from mld import build_mld, marginals_logw, mld_logw

ROUNDS, NOISE = 5, 0.002
# passes (direction, eps, max_states, thr): the first pass decodes every shot; each later pass re-decodes the shots
# where some logical bit still has a probability between thr and 1 - thr. Direction 0 sweeps the mechanisms forward
# in time, 1 backward.
# The second pass used to be (1, 3e-5, 4096, 0.15). This cheaper one is ~1/3 faster overall with no significant
# accuracy difference: compare_settings.py on 100k + 50k fresh shots per basis (seeds 1, 2) gave 349 bits only the
# old setting got right vs 321 only this one got right (sign test: not significant).
MLD_TIERS = [(0, 1e-4, 1024, 0.0), (1, 1e-4, 2048, 0.15)]
MLD_COMPACT_EVERY = 32
MLD_HEXP = 0.5


def _key(circuit: stim.Circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def _tables(circuit: stim.Circuit):
    m = problem_info(circuit)
    # backward sweep: the same decoder on the detectors in reverse order
    return (
        build_mld(m["H"], m["L"], m["p"], hexp=MLD_HEXP),
        build_mld(m["H"][::-1].copy(), m["L"], m["p"], hexp=MLD_HEXP),
    )


_TABLES = {}
for _basis in "XZ":
    _c = apply_circuit_noise(create_circuit(_basis, ROUNDS), p=NOISE)
    _TABLES[_key(_c)] = _tables(_c)
for _g in _TABLES.values():  # compile (or load from numba's cache) the kernel before the clock starts
    for _dir, _eps, _max_states, _ in MLD_TIERS:
        mld_logw(
            _g[_dir],
            np.zeros((2, _g[_dir]["nd"]), bool),
            eps=_eps,
            max_states=_max_states,
            compact_every=MLD_COMPACT_EVERY,
        )


def decode_marginals(g, detector_events: np.ndarray) -> np.ndarray:
    """Per-bit probability of a logical flip, shape (shots, 4)."""
    D = np.asarray(detector_events, dtype=bool)
    events = (np.ascontiguousarray(D), np.ascontiguousarray(D[:, ::-1]))
    logw = None
    for direction, eps, max_states, thr in MLD_TIERS:
        if logw is None:
            logw = mld_logw(
                g[direction],
                events[direction],
                eps=eps,
                max_states=max_states,
                compact_every=MLD_COMPACT_EVERY,
            )
            continue
        sel = np.flatnonzero((np.abs(marginals_logw(logw) - 0.5) < 0.5 - thr).any(1))
        if len(sel):
            logw[sel] = np.maximum(
                logw[sel],
                mld_logw(
                    g[direction],
                    events[direction][sel],
                    eps=eps,
                    max_states=max_states,
                    compact_every=MLD_COMPACT_EVERY,
                ),
            )
    return marginals_logw(logw)


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    g = _TABLES.get(_key(circuit))
    if g is None:
        g = _tables(circuit)
    events = np.asarray(detector_events, dtype=bool)
    assert events.ndim == 2 and events.shape[1] == g[0]["nd"]
    return (decode_marginals(g, events) > 0.5).astype(np.uint8)
    #out, found = lookup(circuit, detector_events)
    ## the table is only trusted for the easiest shots; BP + OSD is more accurate on everything else
    #miss = ~(found & (np.asarray(detector_events).sum(1) <= LUT_MAX_FIRED))
    #if miss.any():
        #m = problem_info(circuit)
        #graph = build_graph(m["H"], m["L"], m["p"])
        #bposd_params = {'max_iter': 8, 'alpha': 0.6, 'always_osd': True, 'damping': 0.1, 'avg_post': True, 'lam': 400, 'lam2': 100, 'coset': 2, 'temp': 0.85, 'refine_thr': 0.3, 'refine_lam': 1000, 'refine_lam2': 300
                        #,'return_stats': False, 'return_classp': False}
        #out[miss] = decode_bp_osd(graph, np.asarray(detector_events)[miss], **bposd_params)
    #return out
