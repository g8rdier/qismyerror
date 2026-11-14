"""Rebuild public circuit assets, score labelled CSVs, or predict unlabelled CSVs.

No organizer package or experimental modules are needed. Outputs never overwrite
existing files. The frozen decoder is not fitted from any labelled samples.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import stim

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_sha256(path):
    """Portable CSV identity: UTF-8 with LF/CRLF input normalized to LF."""
    digest = hashlib.sha256()
    with Path(path).open(encoding="utf-8", newline=None) as f:
        while text := f.read(1 << 20):
            digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def problem_info(circuit):
    """Same unique-column DEM convention as hackathon_2026.problem_info."""
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
        key = (frozenset(detectors), frozenset(observables))
        p = instruction.args_copy()[0]
        if key in columns:
            q = priors[columns[key]]
            priors[columns[key]] = q * (1 - p) + p * (1 - q)
        else:
            columns[key] = len(priors)
            priors.append(p)
    H = np.zeros((dem.num_detectors, len(priors)), np.uint8)
    L = np.zeros((dem.num_observables, len(priors)), np.uint8)
    for (detectors, observables), j in columns.items():
        H[list(detectors), j] = 1
        L[list(observables), j] = 1
    return H, L, np.asarray(priors)


def build(circuits, output):
    if output.exists():
        raise FileExistsError(output)
    arrays = {}
    rng = np.random.default_rng(42071)
    blocks = [(0, 7), (7, 21), (21, 35), (35, 49), (49, 63), (63, 70)]
    shuffled = np.concatenate([rng.permutation(np.arange(a, b)) for a, b in blocks])
    arrays["detector_orders"] = np.stack([np.arange(70), shuffled])
    embedded = {}
    if circuits is None:
        with np.load(HERE / "assets.npz", allow_pickle=False) as z:
            embedded = {b: str(z[f"{b}_circuit"]) for b in "XZ"}
    for basis in "XZ":
        circuit = (stim.Circuit(embedded[basis]) if circuits is None else
                   stim.Circuit.from_file(str(circuits / f"{basis}_circuit.stim")))
        H, L, p = problem_info(circuit)
        arrays.update({f"{basis}_circuit": np.asarray(str(circuit)),
                       f"{basis}_H": H, f"{basis}_L": L, f"{basis}_p": p})
    with output.open("xb") as f:
        np.savez_compressed(f, **arrays)
    print(f"Built {output}: {output.stat().st_size:,} bytes, SHA-256 {sha256(output)}")


def blocks(path, chunk):
    with path.open(newline="") as f:
        header = f.readline().strip().split(",")
        expected = ["id", "basis"] + [f"D{i}" for i in range(70)]
        if header[:72] != expected or header[72:] not in ([], [f"L{i}" for i in range(4)]):
            raise ValueError("expected id,basis,D0..D69[,L0..L3] in column order")
        labelled = len(header) == 76
        while True:
            lines = list(itertools.islice(f, chunk))
            if not lines:
                break
            info = np.loadtxt(lines, delimiter=",", usecols=(0, 1), dtype=str, ndmin=2)
            bits = np.loadtxt(lines, delimiter=",", usecols=range(2, len(header)),
                              dtype=np.uint8, ndmin=2)
            if not np.isin(bits, (0, 1)).all() or not np.isin(info[:, 1], ("X", "Z")).all():
                raise ValueError("CSV must have binary bits and X/Z bases")
            yield info[:, 0], info[:, 1], bits[:, :70], bits[:, 70:] if labelled else None


def load_decoder():
    sys.path.insert(0, str(HERE.parent))
    from mld_ensemble.decoder import decode, _STATES
    circuits = {s["basis"]: stim.Circuit(s["circuit"]) for s in _STATES.values()}
    return decode, circuits


def check_circuits(data, circuits):
    for basis, circuit in circuits.items():
        path = data.parent / f"{basis}_circuit.stim"
        if path.exists() and str(stim.Circuit.from_file(str(path))) != str(circuit):
            raise ValueError(f"{path} does not match this frozen model")


def evaluate(data, output, chunk):
    if output.exists():
        raise FileExistsError(output)
    decode, circuits = load_decoder()
    check_circuits(data, circuits)
    totals = {b: dict(shots=0, wrong=0, sum_loss=0., sum_squared_loss=0., seconds=0.) for b in "XZ"}
    for _, basis, D, labels in blocks(data, chunk):
        if labels is None:
            raise ValueError("evaluation requires L0..L3 labels")
        for b in "XZ":
            selected = basis == b
            if not selected.any():
                continue
            start = time.perf_counter()
            pred = decode(circuits[b], D[selected])
            elapsed = time.perf_counter() - start
            loss = np.mean(pred != labels[selected], axis=1)
            r = totals[b]
            r["shots"] += len(loss)
            r["wrong"] += int(np.count_nonzero(pred != labels[selected]))
            r["sum_loss"] += float(loss.sum())
            r["sum_squared_loss"] += float(np.square(loss).sum())
            r["seconds"] += elapsed
            print(f"{b}: {r['shots']:,} shots, LER {r['wrong'] / (4*r['shots']):.8f}", flush=True)
    n = sum(r["shots"] for r in totals.values())
    mean = sum(r["sum_loss"] for r in totals.values()) / n
    var = max(0., (sum(r["sum_squared_loss"] for r in totals.values()) - n * mean * mean) / (n - 1))
    radius = 1.96 * np.sqrt(var / n)
    report = dict(data_sha256=data_sha256(data), data_checksum_mode="UTF-8 normalized to LF",
                  shots=n, bits=4*n,
                  ler=mean, ci95=[mean-radius, mean+radius], basis=totals,
                  note="Shot-level interval; frozen physics decoder, no fitted weights.")
    args_path = data.parent / "args.json"
    if args_path.exists():
        report["generation"] = json.loads(args_path.read_text())
    output.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


def predict(data, output, chunk):
    if output.exists():
        raise FileExistsError(output)
    decode, circuits = load_decoder()
    check_circuits(data, circuits)
    rows = 0
    with output.open("x", newline="") as f:
        f.write("id,L0,L1,L2,L3\n")
        for ids, bases, D, _ in blocks(data, chunk):
            pred = np.zeros((len(D), 4), np.uint8)
            for basis in "XZ":
                selected = bases == basis
                if selected.any():
                    pred[selected] = decode(circuits[basis], D[selected])
            for identity, p in zip(ids, pred):
                f.write(f"{identity},{p[0]},{p[1]},{p[2]},{p[3]}\n")
            rows += len(D)
            print(f"Predicted {rows:,} rows", flush=True)
    print(json.dumps(dict(rows=rows, input_sha256=data_sha256(data),
                          output_sha256=sha256(output)), indent=2))


def verify(data, prediction):
    model = json.loads((HERE / "model.json").read_text())
    manifest = HERE / "manifest.json"
    checksums = json.loads(manifest.read_text())["sha256"] if manifest.exists() else {}
    for name, expected in checksums.items():
        if sha256(HERE / name) != expected:
            raise ValueError(f"release file checksum mismatch: {name}")
    submission = model.get("submission", {})
    if prediction.resolve() == (HERE / "results.csv").resolve() and submission:
        if data_sha256(data) != submission["input_sha256"] or sha256(prediction) != submission["output_sha256"]:
            raise ValueError("input/prediction checksum does not match model.json")
    rows = 0
    with prediction.open(newline="") as f:
        if f.readline().strip() != "id,L0,L1,L2,L3":
            raise ValueError("wrong prediction header")
        for ids, _, _, _ in blocks(data, 100000):
            for identity in ids:
                fields = f.readline().strip().split(",")
                if len(fields) != 5 or fields[0] != identity or not set(fields[1:]) <= {"0", "1"}:
                    raise ValueError(f"prediction mismatch at row {rows}")
                rows += 1
        if f.readline():
            raise ValueError("extra prediction rows")
    print(f"Verified {rows:,} binary prediction rows in exact input order.")


def manifest():
    """Normalize release text to LF and fingerprint it; no fitting occurs."""
    output = HERE / "manifest.json"
    if output.exists():
        raise FileExistsError(output)
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p != output)
    for path in files:
        if path.suffix in (".py", ".json", ".md", ".txt", ".csv") or path.name in ("LICENSE", ".gitattributes"):
            before = path.read_bytes()
            normalized = before.replace(b"\r\n", b"\n")
            if normalized != before:
                path.write_bytes(normalized)
    report = {"sha256": {p.name: sha256(p) for p in files}}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(f"Recorded {len(files)} release file checksums.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, help="default: available Numba threads")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--circuits", type=Path, help="default: public circuit texts embedded in assets.npz")
    b.add_argument("--out", type=Path, required=True)
    sub.add_parser("manifest")
    for name in ("evaluate", "predict", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name != "verify":
            p.add_argument("--chunk", type=int, default=100000)
    args = parser.parse_args()
    import numba
    if args.threads is not None:
        numba.set_num_threads(args.threads)
    if args.command == "build":
        build(args.circuits, args.out)
    elif args.command == "evaluate":
        evaluate(args.data, args.out, args.chunk)
    elif args.command == "predict":
        predict(args.data, args.out, args.chunk)
    elif args.command == "verify":
        verify(args.data, args.out)
    else:
        manifest()


if __name__ == "__main__":
    main()
