"""Save exact intermediate outputs to compare decoder revisions.

Run this script separately for the reference and candidate decoder folders,
using the same thread count. Compare every array in the resulting NPZ files
with numpy.testing.assert_array_equal. Generated shots use fixed Stim seeds;
no test labels or saved predictions are loaded.
"""
import argparse
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--decoder', required=True)
p.add_argument('--threads', type=int, required=True)
p.add_argument('--out', required=True)
p.add_argument('--shots', type=int, default=2048)
a = p.parse_args()
for key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = str(a.threads)
sys.path.insert(0, a.decoder)
import numpy as np
import numba
import stim
import decoder
import mld

# The optimized guard is valid for every mechanism of both circuit sweeps.
for tables in decoder._TABLES.values():
    for table in tables:
        assert np.all(table['cv0'][~table['hasv0']] == 0)

arrays = {}
basis = None
record = False
stage = 0
original_sweeps = decoder.mld_logw_many
original_marginals = decoder.decode_marginals

def traced_sweeps(sweeps, **kw):
    global stage
    result = original_sweeps(sweeps, **kw)
    if record:
        weights = result[0] if kw.get('return_costs') else result
        for i, w in enumerate(weights):
            arrays[f'{basis}_stage{stage}_sweep{i}'] = w.copy()
        stage += 1
    return result

def traced_marginals(g, events):
    result = original_marginals(g, events)
    if record:
        arrays[f'{basis}_marginals'] = result.copy()
    return result

decoder.mld_logw_many = traced_sweeps
decoder.decode_marginals = traced_marginals

for basis in 'XZ':
    circuit = stim.Circuit.from_file(str(Path(a.decoder) / 'circuits' / f'{basis}_circuit.stim'))
    events = circuit.compile_detector_sampler(seed=202611140 + ord(basis)).sample(a.shots)
    arrays[f'{basis}_events'] = events
    stage = 0
    record = True
    pred = decoder.decode(circuit, events)
    record = False
    arrays[f'{basis}_predictions'] = pred
    np.testing.assert_array_equal(pred[::-1], decoder.decode(circuit, events[::-1]))
    np.testing.assert_array_equal(pred, np.concatenate([
        decoder.decode(circuit, events[:a.shots//2]),
        decoder.decode(circuit, events[a.shots//2:])]))
    np.testing.assert_array_equal(decoder.decode(circuit, events[:0]), np.empty((0, 4), np.uint8))
    duplicates = np.repeat(events[:8], 7, axis=0)
    arrays[f'{basis}_duplicates'] = decoder.decode(circuit, duplicates)
    np.testing.assert_array_equal(arrays[f'{basis}_duplicates'], np.repeat(pred[:8], 7, axis=0))

# Exact integer-helper comparison covers zero, every single-bit position, and
# random words containing multiple set bits. Both baseline and candidate run it.
words = np.concatenate([np.array([0], np.uint64), np.uint64(1) << np.arange(64, dtype=np.uint64),
                        np.random.default_rng(91).integers(0, 2**64-1, 10000, dtype=np.uint64)])
@numba.njit
def bit_indices(words):
    return np.array([mld._ctz(w) for w in words])
indices = bit_indices(words)
expected = np.array([(int(w) & -int(w)).bit_length()-1 if w else 0 for w in words])
np.testing.assert_array_equal(indices, expected)
arrays['bit_indices'] = indices
np.savez(a.out, **arrays)
print(f'Passed generated, reversed, partitioned, duplicate and empty shots; {len(words)} exact bit-index checks; saved {len(arrays)} arrays.', flush=True)
