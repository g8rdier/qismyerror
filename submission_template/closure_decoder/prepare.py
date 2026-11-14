"""Optional reproducibility and prediction utility; not imported by decode().

Examples, from the repository root:
  python submission_template/closure_decoder/prepare.py predict --data data/test/test.csv
  python submission_template/closure_decoder/prepare.py verify --data data/test/test.csv
  python submission_template/closure_decoder/prepare.py train --circuits data/train --out NEW_DIR

No training is required to use the included assets. Training samples are
generated directly with Stim; this utility never trains on a test CSV.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
HEADER = ["id", "L0", "L1", "L2", "L3"]
RUNTIME_FILES = ["__init__.py", "decoder.py", "bp_osd.py", "closure.py", "candidates.py"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(path):
    """Git may check CSV/Python text out as LF or CRLF; hash the same text."""
    h = hashlib.sha256()
    with Path(path).open("r", encoding="utf-8", newline=None) as f:
        for chunk in iter(lambda: f.read(1 << 20), ""):
            h.update(chunk.encode("utf-8"))
    return h.hexdigest()


def configuration_hash(model):
    payload = {"parameters": model["parameters"], "lut_max_fired": model["lut_max_fired"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {name: text_sha256(HERE / name) for name in RUNTIME_FILES + ["prepare.py"]}


def problem_info(circuit):
    """Extract the public independent detector-error model, merging equal effects."""
    import numpy as np

    dem = circuit.detector_error_model()
    columns, priors = {}, []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        detectors, observables = set(), set()
        for target in ins.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
        key = (frozenset(detectors), frozenset(observables))
        p = ins.args_copy()[0]
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


def _merge(keys, counts, ones):
    import numpy as np

    unique, inverse = np.unique(keys, return_inverse=True)
    n = np.bincount(inverse, weights=counts, minlength=len(unique)).astype(np.int64)
    o = np.stack([np.bincount(inverse, weights=ones[:, j], minlength=len(unique))
                  for j in range(4)], axis=1).astype(np.int64)
    return unique, n, o


def train(args):
    """Streaming head-table fit: no giant CSVs, no change to shipped assets."""
    import numpy as np
    import stim

    if args.shots < 1 or args.chunk < 1:
        raise ValueError("shots and chunk must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite training output {out}")
    out.mkdir(parents=True)
    manifest = {"shots_per_basis": args.shots, "chunk": args.chunk, "max_fired": 4,
                "stim_version": stim.__version__, "numpy_version": np.__version__,
                "generator_sha256": sha256(__file__), "bases": {}}
    for k, basis in enumerate("XZ"):
        c = stim.Circuit.from_file(str(Path(args.circuits) / f"{basis}_circuit.stim"))
        if c.num_detectors != 70 or c.num_observables != 4:
            raise ValueError("this release is for the 70-detector, four-observable circuits")
        seed = args.seed + k
        sampler = c.compile_detector_sampler(seed=seed)
        keys = np.empty(0, dtype="V9")
        counts = np.empty(0, np.int64)
        ones = np.empty((0, 4), np.int64)
        done, retained = 0, 0
        start = time.perf_counter()
        while done < args.shots:
            n = min(args.chunk, args.shots - done)
            D, O = sampler.sample(n, separate_observables=True, bit_packed=True)
            keep = np.bitwise_count(D).sum(1) <= 4
            q = np.ascontiguousarray(D[keep]).view("V9").ravel()
            o = np.unpackbits(O[keep], axis=1, count=4, bitorder="little").astype(np.int64)
            keys, counts, ones = _merge(
                np.concatenate([keys, q]),
                np.concatenate([counts, np.ones(len(q), np.int64)]),
                np.concatenate([ones, o]))
            done += n
            retained += len(q)
            if done % (5 * args.chunk) == 0 or done == args.shots:
                print(f"{basis}: {done:,}/{args.shots:,}, {len(keys):,} head patterns",
                      flush=True)
        bits = (2 * ones > counts[:, None]).astype(np.uint8)
        prediction = np.packbits(bits, axis=1, bitorder="little")[:, 0]
        path = out / f"lut_{basis}.npz"
        np.savez_compressed(path, keys=keys.view(np.uint8).reshape(-1, 9),
                            predict=prediction, counts=counts, ones=ones)
        manifest["bases"][basis] = {
            "seed": seed, "circuit_sha256": hashlib.sha256(str(c).encode()).hexdigest(),
            "patterns": len(keys), "retained_shots": retained, "file_sha256": sha256(path),
            "seconds": time.perf_counter() - start}
        (out / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Training complete: {out}", flush=True)


def build(args):
    """Package already trained tables and public circuit physics in one NPZ."""
    import numpy as np
    import stim

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite assets {out}")
    payload, info = {}, {}
    for basis in "XZ":
        c = stim.Circuit.from_file(str(Path(args.circuits) / f"{basis}_circuit.stim"))
        H, L, p = problem_info(c)
        if H.shape[0] != 70 or L.shape[0] != 4:
            raise ValueError("unexpected circuit dimensions")
        with np.load(Path(args.tables) / f"lut_{basis}.npz", allow_pickle=False) as z:
            keys = np.ascontiguousarray(z["keys"]).reshape(-1, 9)
            labels = z["predict"]
            keep = np.bitwise_count(keys).sum(1) <= 4
            keys, labels = keys[keep], labels[keep]
        void = np.ascontiguousarray(keys).view("V9").ravel()
        order = np.argsort(void)
        if len(np.unique(void)) != len(void):
            raise ValueError("duplicate lookup keys")
        payload.update({f"{basis}_H": H, f"{basis}_L": L, f"{basis}_p": p,
                        f"{basis}_circuit": np.array(str(c)),
                        f"{basis}_keys": keys[order], f"{basis}_predict": labels[order]})
        info[basis] = {"circuit_sha256": hashlib.sha256(str(c).encode()).hexdigest(),
                       "patterns": len(keys), "mechanisms": H.shape[1],
                       "H_sha256": hashlib.sha256(H.tobytes()).hexdigest(),
                       "L_sha256": hashlib.sha256(L.tobytes()).hexdigest(),
                       "p_sha256": hashlib.sha256(p.tobytes()).hexdigest()}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(json.dumps({"assets": str(out), "sha256": sha256(out),
                      "bytes": out.stat().st_size, "bases": info}, indent=2))


def read_data(path, labels=False):
    import numpy as np

    path = Path(path)
    with path.open(newline="") as f:
        header = next(csv.reader(f))
    names = [f"D{i}" for i in range(70)]
    if any(name not in header for name in names):
        raise ValueError("expected detector columns D0 through D69")
    text = np.loadtxt(path, delimiter=",", skiprows=1,
                      usecols=(header.index("id"), header.index("basis")), dtype=str, ndmin=2)
    D = np.loadtxt(path, delimiter=",", skiprows=1,
                   usecols=[header.index(name) for name in names], dtype=np.int16, ndmin=2)
    if not np.isin(D, (0, 1)).all() or not np.isin(text[:, 1], ["X", "Z"]).all():
        raise ValueError("invalid detector values or bases")
    result = {"id": text[:, 0], "basis": text[:, 1], "D": D.astype(bool)}
    if labels:
        result["O"] = np.loadtxt(
            path, delimiter=",", skiprows=1, usecols=[header.index(f"L{i}") for i in range(4)],
            dtype=np.uint8, ndmin=2)
    return result


def _submission(ids, prediction):
    import numpy as np

    if prediction.shape != (len(ids), 4) or not np.isin(prediction, (0, 1)).all():
        raise ValueError("invalid prediction shape or values")
    f = io.StringIO(newline="")
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(HEADER)
    for shot_id, row in zip(ids, prediction):
        writer.writerow([str(shot_id)] + [int(x) for x in row])
    return f.getvalue().encode("utf-8")


def _check_csv(path, data, sample=None):
    import numpy as np

    with Path(path).open(newline="") as f:
        reader = csv.reader(f)
        if next(reader) != HEADER:
            raise ValueError("wrong submission header")
        rows = list(reader)
    if len(rows) != len(data["id"]):
        raise ValueError("submission row count does not match input")
    if any(len(row) != 5 or any(v not in ("0", "1") for v in row[1:]) for row in rows):
        raise ValueError("non-binary/malformed submission")
    ids = np.asarray([row[0] for row in rows])
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, data["id"]):
        raise ValueError("submission IDs are duplicated or not in input row order")
    if sample:
        with Path(sample).open(newline="") as f:
            reader = csv.reader(f)
            if next(reader) != HEADER:
                raise ValueError("sample submission has a different header")
            sample_ids = np.asarray([row[0] for row in reader])
        if not np.array_equal(sample_ids, ids):
            raise ValueError("submission IDs/order differ from sample submission")


def predict(args):
    import numpy as np
    import stim

    data = read_data(args.data)
    with Path(args.data).open(newline="") as f:
        header = next(csv.reader(f))
    if any(f"L{i}" in header for i in range(4)):
        raise ValueError("predict is for unlabelled submission inputs, not labelled evaluation")
    if args.runs < 2:
        raise ValueError("use at least two runs to verify deterministic submission generation")
    if __package__:
        from . import decoder
    else:
        import decoder

    circuits = {b: stim.Circuit.from_file(str(Path(args.data).parent / f"{b}_circuit.stim"))
                for b in "XZ"}
    first_bytes, times = None, []
    for run in range(args.runs):
        out = np.empty((len(data["D"]), 4), np.uint8)
        seconds = 0.
        for basis in "XZ":
            rows = np.flatnonzero(data["basis"] == basis)
            start = time.perf_counter()
            pp = decoder.decode(circuits[basis], data["D"][rows])
            if pp.shape != (len(rows), 4) or not np.isin(pp, (0, 1)).all():
                raise ValueError("decoder violated its output contract")
            seconds += time.perf_counter() - start
            out[rows] = pp
        payload = _submission(data["id"], out)
        if first_bytes is None:
            first_bytes = payload
        elif payload != first_bytes:
            raise AssertionError("submission bytes differ between identical runs")
        times.append(seconds)
        print(f"run {run + 1}: {seconds:.3f}s, sha256 "
              f"{hashlib.sha256(payload).hexdigest()}", flush=True)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != first_bytes:
        raise SystemExit("refusing to overwrite a different submission; choose a new --out path")
    destination.write_bytes(first_bytes)
    _check_csv(destination, data, args.sample)
    model_path = HERE / "model.json"
    model = json.loads(model_path.read_text())
    model["submission"] = {
        "file": destination.name, "rows": len(data["id"]), "header": HEADER,
        "input_sha256": text_sha256(args.data), "input_raw_sha256": sha256(args.data),
        "text_hash_normalization": "UTF-8 text with universal LF newlines",
        "results_sha256": sha256(destination),
        "assets_sha256": sha256(HERE / "assets.npz"),
        "configuration_sha256": configuration_hash(model),
        "source_sha256": source_hashes(), "deterministic_runs": args.runs,
        "local_seconds": times, "test_ler": None,
        "note": "Unlabelled test predictions; no test LER measured."}
    model_path.write_text(json.dumps(model, indent=2) + "\n")
    print(f"Wrote and checked {len(data['id']):,} predictions: {destination}")


def verify(args):
    model = json.loads((HERE / "model.json").read_text())
    if sha256(HERE / "assets.npz") != model["assets_sha256"]:
        raise AssertionError("assets checksum differs from frozen model")
    if not args.data:
        print("Model asset checksum verified")
        return
    data = read_data(args.data)
    _check_csv(args.submission, data, args.sample)
    saved = model["submission"]
    for actual, expected in (
        (text_sha256(args.data), saved["input_sha256"]),
        (sha256(args.submission), saved["results_sha256"]),
        (sha256(HERE / "assets.npz"), saved["assets_sha256"]),
        (configuration_hash(model), saved["configuration_sha256"]),
        (source_hashes(), saved["source_sha256"]),
    ):
        if actual != expected:
            raise AssertionError("submission/model/source identity check failed")
    print("Submission format, row order, assets, configuration and source checksums verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--circuits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shots", type=int, default=50_000_000)
    p.add_argument("--chunk", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=961001)
    p.set_defaults(fn=train)
    p = sub.add_parser("build")
    p.add_argument("--circuits", type=Path, required=True)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=build)
    p = sub.add_parser("predict")
    p.add_argument("--data", type=Path, default=Path("data/test/test.csv"))
    p.add_argument("--sample", type=Path, default=Path("sample_submission.csv"))
    p.add_argument("--out", type=Path, default=HERE / "results.csv")
    p.add_argument("--runs", type=int, default=2)
    p.set_defaults(fn=predict)
    p = sub.add_parser("verify")
    p.add_argument("--data", type=Path)
    p.add_argument("--sample", type=Path, default=Path("sample_submission.csv"))
    p.add_argument("--submission", type=Path, default=HERE / "results.csv")
    p.set_defaults(fn=verify)
    args = parser.parse_args()
    if args.threads:
        if args.threads < 1:
            parser.error("threads must be positive")
        for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(args.threads)
    args.fn(args)


if __name__ == "__main__":
    main()
