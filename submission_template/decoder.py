"""Near-maximum-likelihood decoder for the code check.

The top 10 teams hand in this folder: decoder.py, requirements.txt and any files decode() loads, together with
the code that created them. We call decode() once per basis on that basis's 100,000 shots in data/kaggle_private and time the two
calls (see "Handing in your code" in the README). Importing the decoder, and anything it builds when imported, is
not timed.

Every shot is decoded by the pruned dynamic programme in mld.py, which computes the weight of each of the 16 logical
outcomes and predicts each logical bit by its marginal (the optimal decision for the per-bit LER).

Pruning only removes terms from an outcome's weight, so every sweep gives a lower bound on it, and sweeps are
combined by keeping the largest bound per outcome. A sweep over the error mechanisms forward in time and one
backward in time prune different error vectors, so their combination is closer to the exact weights than one sweep
with a tighter threshold that costs the same.

Their disagreement also shows where the pruning mattered. Every shot gets a forward and a backward sweep with a
loose threshold. A shot is swept again in both directions with a tighter threshold if the sweeps (or their
combination) differ on some logical bit's probability by an amount that matters for the decision, that is, relative
to how far that probability is from 1/2; this is repeated once more with a much tighter threshold. About 14% of the
shots get the second pair of sweeps and 2.5% the third. Against an exact (unpruned) decoder the remaining loss is
below one bit in a million.

At import we precompute the decoder tables for the competition's two circuits (circuits/, copied from the Data tab)
and compile the numba kernel, so decode() only has to look the tables up by a hash of the circuit text. A circuit
that is not one of these is handled too, by building its tables inside decode(). The folder needs nothing outside
itself and requirements.txt.
"""
import hashlib
import os

import numpy as np
import stim

from mld import build_mld, marginals_logw, mld_logw, mld_logw_many

HERE = os.path.dirname(os.path.abspath(__file__))
# stages (sweeps, thr): each sweep is (direction, eps, max_states), direction 0 forward in time, 1 backward. The
# first stage decodes every shot. A shot goes on to the next stage if, for some logical bit, the probabilities from
# the stage's sweeps, the combined result before the stage and the combined result after it differ by more than
# thr * (distance of the combined probability from 1/2 + MARGIN_FLOOR): the tighter sweeps go where the pruning
# error could still change a decision.
MLD_STAGES = [
    (((0, 3e-4, 4096), (1, 3e-4, 4096)), 0.02),
    (((0, 3e-5, 4096), (1, 3e-5, 4096)), 0.3),
    (((0, 3e-6, 8192), (1, 3e-6, 8192)), 0.0),
]
MARGIN_FLOOR = 0.01
# Bits whose probability is within TIE_BAND of 1/2 are coin flips: predicting the less likely value there costs at most
# 2 * TIE_BAND expected errors per bit. TIE_BAND = 0 is the plain maximum-a-posteriori decision.
TIE_BAND = 0.01


def problem_info(circuit: stim.Circuit):
    """H (detectors x mechanisms), L (observables x mechanisms) and p (probability of each mechanism). Error
    mechanisms with the same detectors and observables are merged into one column."""
    dem = circuit.detector_error_model()
    columns, priors = {}, []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
        key, p = (frozenset(detectors), frozenset(observables)), instruction.args_copy()[0]
        if key in columns:  # same effect as an earlier mechanism: either one fires, not both
            q = priors[columns[key]]
            priors[columns[key]] = q * (1 - p) + p * (1 - q)
        else:
            columns[key] = len(priors)
            priors.append(p)
    H = np.zeros((dem.num_detectors, len(priors)), dtype=np.uint8)
    L = np.zeros((dem.num_observables, len(priors)), dtype=np.uint8)
    for (detectors, observables), j in columns.items():
        H[list(detectors), j] = 1
        L[list(observables), j] = 1
    return H, L, np.array(priors)


def _key(circuit: stim.Circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def _tables(circuit: stim.Circuit):
    H, L, p = problem_info(circuit)
    # backward sweep: the same decoder on the detectors in reverse order
    return build_mld(H, L, p), build_mld(H[::-1].copy(), L, p)


_TABLES = {}
for _basis in "XZ":
    _c = stim.Circuit.from_file(os.path.join(HERE, "circuits", f"{_basis}_circuit.stim"))
    _TABLES[_key(_c)] = _tables(_c)
for _g in _TABLES.values():  # compile (or load from numba's cache) the kernel before the clock starts
    for _sweeps, _ in MLD_STAGES:
        for _dir, _eps, _max_states in _sweeps:
            mld_logw(_g[_dir], np.zeros((2, _g[_dir]["nd"]), bool), eps=_eps, max_states=_max_states)


def decode_marginals(g, detector_events: np.ndarray) -> np.ndarray:
    """Per-bit probability of a logical flip, shape (shots, 4)."""
    D = np.asarray(detector_events, dtype=bool)
    events = (np.ascontiguousarray(D), np.ascontiguousarray(D[:, ::-1]))
    logw = np.full((len(D), 16), -np.inf)  # best lower bound so far on the log weight of each outcome
    sel = np.arange(len(D))
    cost = None  # peak state counts of the previous stage, only used to start the hardest shots first
    for stage, (sweeps, thr) in enumerate(MLD_STAGES):
        if len(sel) == 0:
            break
        views = [marginals_logw(logw[sel])] if stage else []
        results, cost = mld_logw_many([(g[direction], events[direction][sel], eps, max_states) for direction, eps, max_states in sweeps],
                                      priority=cost, return_costs=True)
        for lw in results:
            views.append(marginals_logw(lw))
            logw[sel] = np.maximum(logw[sel], lw)
        views.append(marginals_logw(logw[sel]))
        spread = np.max(views, 0) - np.min(views, 0)
        escalate = (spread / (np.abs(views[-1] - 0.5) + MARGIN_FLOOR)).max(1) > thr
        sel, cost = sel[escalate], cost[escalate]
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
    m = decode_marginals(g, events)
    pred = m > 0.5
    if TIE_BAND > 0:
        pred ^= np.abs(m - 0.5) < TIE_BAND
    return pred.astype(np.uint8)
