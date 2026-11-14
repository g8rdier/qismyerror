"""Frozen physics-based MLD ensemble. No fitting, CSV reads or network access.

Assets and the Numba kernel load/compile at import, before timed decode calls.
The package is self-contained and may be copied without the research folders.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
if "mld_ensemble" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "mld_ensemble", HERE / "__init__.py", submodule_search_locations=[str(HERE)])
    _package = importlib.util.module_from_spec(_spec)
    sys.modules["mld_ensemble"] = _package
    _spec.loader.exec_module(_package)
elif Path(sys.modules["mld_ensemble"].__file__).resolve().parent != HERE:
    raise ImportError("a different mld_ensemble package is loaded; use a fresh process")

from mld_ensemble.kernel import build_mld, marginals_logw, mld_logw

MODEL = json.loads((HERE / "model.json").read_text())
PARAMETERS = MODEL["parameters"]
ASSETS = HERE / "assets.npz"
if hashlib.sha256(ASSETS.read_bytes()).hexdigest() != MODEL["assets_sha256"]:
    raise ValueError("assets.npz does not match model.json")


def _make_states(path):
    states = {}
    with np.load(path, allow_pickle=False) as z:
        orders = z["detector_orders"].copy()
        if orders.shape != (2, 70) or not all(
                np.array_equal(np.sort(order), np.arange(70)) for order in orders):
            raise ValueError("invalid detector permutations")
        for basis in "XZ":
            text = str(z[f"{basis}_circuit"])
            H, L, p = z[f"{basis}_H"], z[f"{basis}_L"], z[f"{basis}_p"]
            if H.shape[0] != 70 or L.shape != (4, H.shape[1]) or p.shape != (H.shape[1],):
                raise ValueError("invalid model dimensions")
            if not np.isin(H, (0, 1)).all() or not np.isin(L, (0, 1)).all():
                raise ValueError("model matrices must be binary")
            if not ((p > 0) & (p < .5)).all():
                raise ValueError("invalid mechanism probabilities")
            tables = [
                [build_mld(H[order[::direction]].copy(), L, p,
                           hexp=PARAMETERS["heuristic_exponent"])
                 for direction in (1, -1)]
                for order in orders
            ]
            key = hashlib.sha256(text.encode()).hexdigest()
            states[key] = dict(basis=basis, circuit=text, orders=orders, tables=tables)
    return states


def _component(state, D, component):
    """Absolute outcome masses; reverse pass retains the larger per-class bound."""
    weights = None
    order = state["orders"][component]
    for direction, eps, max_states, threshold in PARAMETERS["tiers"]:
        selected = np.arange(len(D)) if weights is None else np.flatnonzero(
            (np.abs(marginals_logw(weights) - .5) < .5 - threshold).any(1))
        if not len(selected):
            continue
        events = D[selected][:, order[::(-1 if direction else 1)]]
        extra = mld_logw(state["tables"][component][direction], events,
                         eps=eps, max_states=max_states,
                         compact_every=PARAMETERS["compact_every"])
        if weights is None:
            weights = extra
        else:
            weights[selected] = np.maximum(weights[selected], extra)
    return weights


def _unique(D):
    packed = np.ascontiguousarray(np.packbits(D, axis=1, bitorder="little"))
    _, first, inverse = np.unique(packed.view("V9").ravel(),
                                  return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return D[first[order]], remap[inverse]


def _decode_state(state, D):
    if not len(D):
        return np.empty((0, 4), np.uint8)
    unique, inverse = _unique(D)
    weights = np.maximum(_component(state, unique, 0), _component(state, unique, 1))
    return (marginals_logw(weights) > .5).astype(np.uint8)[inverse]


_STATES = _make_states(ASSETS)
for _state in _STATES.values():
    for _tables in _state["tables"]:
        for _graph in _tables:
            mld_logw(_graph, np.zeros((1, 70), bool),
                     eps=1e-4, max_states=1024,
                     compact_every=PARAMETERS["compact_every"])


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """(shots, 70) binary detector events -> (shots, 4) uint8 logical flips."""
    raw = np.asarray(detector_events)
    if raw.ndim != 2 or raw.shape[1] != 70:
        raise ValueError("expected detector_events with shape (shots, 70)")
    if not np.isin(raw, (0, 1)).all():
        raise ValueError("detector events must be binary")
    state = _STATES.get(hashlib.sha256(str(circuit).encode()).hexdigest())
    if state is None:
        raise ValueError("unrecognized circuit; assets contain the competition X/Z models")
    return _decode_state(state, np.asarray(raw, dtype=bool))
