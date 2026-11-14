"""Time a decoder and compare every output bit with a frozen submission.

Run with the hand-in environment's Python. This tool needs only the hand-in's
NumPy/Stim/Numba dependencies; it does not import the project package or labels.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--data', type=Path, required=True, help='CSV with id, basis, D0..D69')
    parser.add_argument('--submission', type=Path, required=True, help='Frozen selected submission')
    parser.add_argument('--threads', type=int, default=48)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1:
        parser.error('--runs and --threads must be positive')
    for name in ('OMP_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(args.threads)

    import numpy as np
    import stim

    sys.path.insert(0, str(args.decoder.resolve()))
    started = time.perf_counter()
    from decoder import decode
    import_seconds = time.perf_counter() - started

    with args.data.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    ids = [row[header.index('id')] for row in rows]
    bases = np.array([row[header.index('basis')] for row in rows])
    columns = [header.index(f'D{i}') for i in range(70)]
    events = np.array([[int(row[i]) for i in columns] for row in rows], dtype=bool)
    del rows
    with args.submission.open(newline='') as f:
        reader = csv.reader(f)
        if next(reader) != ['id', 'L0', 'L1', 'L2', 'L3']:
            raise ValueError('Wrong submission header')
        rows = list(reader)
    if ids != [row[0] for row in rows]:
        raise ValueError('Input and selected submission IDs/order differ')
    expected = np.array([row[1:] for row in rows], dtype=np.uint8)
    del rows
    if expected.shape != (len(events), 4) or not np.isin(expected, (0, 1)).all():
        raise ValueError('Invalid selected submission predictions')
    if not np.isin(bases, ('X', 'Z')).all():
        raise ValueError('Unknown basis')
    selected = {b: np.flatnonzero(bases == b) for b in 'XZ'}
    data = {b: events[selected[b]] for b in 'XZ'}
    circuits = {b: stim.Circuit.from_file(str(args.data.parent / f'{b}_circuit.stim')) for b in 'XZ'}
    records = []
    for run in range(args.runs):
        actual = np.empty_like(expected)
        seconds = {}
        for b in 'XZ':
            started = time.perf_counter()
            pred = decode(circuits[b], data[b])
            seconds[b] = time.perf_counter() - started
            pred = np.asarray(pred)
            if pred.shape != (len(data[b]), 4) or not np.isin(pred, (0, 1)).all():
                raise ValueError(f'Invalid decoder output for {b}')
            actual[selected[b]] = pred
            print(json.dumps({'run': run + 1, 'basis': b, 'seconds': seconds[b]}), flush=True)
        changed = int(np.count_nonzero(actual != expected))
        record = {'run': run + 1, 'seconds': seconds, 'total_seconds': sum(seconds.values()), 'changed_bits': changed}
        records.append(record)
        print(json.dumps(record), flush=True)
        if changed:
            raise AssertionError(f'{changed} prediction bits differ from the selected submission')

    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['id', 'L0', 'L1', 'L2', 'L3'])
    writer.writerows([identity, *map(int, p)] for identity, p in zip(ids, actual))
    output_bytes = output.getvalue().encode()
    selected_bytes = args.submission.read_bytes()
    # The competition compares predictions; this selected baseline also has
    # canonical LF formatting, so require identical CSV bytes as a stronger check.
    if output_bytes != selected_bytes:
        raise AssertionError('CSV bytes differ from the selected submission')
    source_files = ['decoder.py', 'mld.py', 'requirements.txt',
                    'circuits/X_circuit.stim', 'circuits/Z_circuit.stim']
    report = {
        'selected_submission': args.submission.name,
        'selected_submission_sha256': hashlib.sha256(selected_bytes).hexdigest(),
        'input_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'shots': len(events), 'bits': int(expected.size),
        'threads': args.threads, 'runs': records,
        'median_seconds': statistics.median(r['total_seconds'] for r in records),
        'import_seconds_excluded': import_seconds,
        'byte_identical_csv': True,
        'source_sha256': {name: hashlib.sha256((args.decoder / name).read_bytes()).hexdigest() for name in source_files},
        'python': sys.version,
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'stim', 'numba', 'llvmlite')},
        'official_hardware_timing': False,
    }
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'median_seconds': report['median_seconds'], 'byte_identical_csv': True, 'report': str(args.report)}))


if __name__ == '__main__':
    main()
