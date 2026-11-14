"""Independent ordering and state-copy checks for runtime-only changes."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mld import _argsort_states


@njit(cache=True)
def _reference_order(scores, keep):
    return np.argsort(-scores)[:keep]


class SortTests(unittest.TestCase):
    def check_order(self, scores, keep):
        before = scores.view(np.uint64).copy()
        actual = _argsort_states(scores, keep)
        np.testing.assert_array_equal(actual, _reference_order(scores, keep))
        np.testing.assert_array_equal(scores.view(np.uint64), before)

    def test_exhaustive_small_ties_and_prefixes(self):
        for n in range(9):
            for row in itertools.product((-1., 0., 1.), repeat=n):
                scores = np.array(row, dtype=np.float64)
                for keep in range(n + 1):
                    self.check_order(scores, keep)

    def test_exhaustive_first_quicksort_partition(self):
        for row in itertools.product((-1., 0.), repeat=16):
            scores = np.array(row)
            for keep in (8, 16):
                self.check_order(scores, keep)

    def test_large_random_sorted_and_special_values(self):
        rng = np.random.default_rng(38102)
        for n in (0, 1, 15, 16, 31, 32, 100, 1024, 4096, 8193, 16384, 65537):
            cases = [rng.normal(size=n), rng.integers(-5, 6, n).astype(float),
                     np.zeros(n), np.arange(n, dtype=float), -np.arange(n, dtype=float)]
            bits = rng.integers(0, 2**64 - 1, n, dtype=np.uint64)
            cases.append(bits.view(np.float64))
            if n >= 10:
                special = rng.normal(size=n)
                special[:10] = [-np.inf, np.inf, -0., 0., np.nan, -np.nan,
                                np.nextafter(0., 1.), -np.nextafter(0., 1.),
                                np.finfo(float).max, -np.finfo(float).max]
                cases.append(special)
            for scores in cases:
                for keep in (0, n // 2, n):
                    self.check_order(scores, keep)



if __name__ == '__main__':
    unittest.main()
