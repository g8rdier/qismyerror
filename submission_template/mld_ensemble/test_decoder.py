"""Exhaustive kernel checks and independent, repeatable release API tests."""
from pathlib import Path
import sys
import tempfile
import unittest

import numba
import numpy as np
import stim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mld_ensemble.kernel import _ctz, build_mld, marginals_logw, mld_logw


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        rng = np.random.default_rng(42010)
        cls.H = np.zeros((70, 9), np.uint8)
        cls.H[[0, 2, 63, 64, 69]] = rng.integers(0, 2, (5, 9), dtype=np.uint8)
        cls.H[:, 0] = 0
        cls.L = rng.integers(0, 2, (4, 9), dtype=np.uint8)
        cls.p = rng.uniform(.02, .2, 9)
        E = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(np.uint8)
        cls.D, inv = np.unique((E @ cls.H.T) % 2, axis=0, return_inverse=True)
        logical = ((E @ cls.L.T) % 2) @ (1 << np.arange(4))
        cls.exact = np.zeros((len(cls.D), 16))
        np.add.at(cls.exact, (inv, logical), np.prod(np.where(E, cls.p/(1-cls.p), 1), axis=1))

    def test_all_bit_indices(self):
        for i in range(64):
            self.assertEqual(_ctz(np.uint64(1) << np.uint64(i)), i)

    def test_exact_forward_reverse(self):
        for direction in (1, -1):
            g = build_mld(self.H[::direction].copy(), self.L, self.p)
            actual = np.exp(mld_logw(g, self.D[:, ::direction], eps=1e-100, max_states=1024))
            np.testing.assert_allclose(actual, self.exact, atol=1e-13, rtol=2e-12)

    def test_pruning_is_a_lower_bound(self):
        g = build_mld(self.H, self.L, self.p)
        actual = np.exp(mld_logw(g, self.D, eps=.1, max_states=8))
        self.assertTrue(np.all(actual <= self.exact + 1e-12))

    def test_impossible_and_empty(self):
        g = build_mld(self.H, self.L, self.p)
        D = np.zeros((1, 70), bool)
        D[0, 1] = True
        np.testing.assert_array_equal(marginals_logw(mld_logw(g, D)), np.full((1, 4), .5))
        self.assertEqual(mld_logw(g, D[:0]).shape, (0, 16))

    def test_portable_csv_checksum(self):
        from mld_ensemble.prepare import data_sha256
        approved = Path("C:/Users/nicho/AppData/Local/Temp/opencode")
        with tempfile.TemporaryDirectory(prefix="mld_checksum_", dir=approved if approved.exists() else None) as t:
            a, b = Path(t) / "lf.csv", Path(t) / "crlf.csv"
            a.write_bytes(b"id,L0\nexample,0\n")
            b.write_bytes(b"id,L0\r\nexample,0\r\n")
            self.assertEqual(data_sha256(a), data_sha256(b))


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        from mld_ensemble.decoder import decode, _STATES
        cls.decode = staticmethod(decode)
        cls.circuits = [stim.Circuit(s["circuit"]) for s in _STATES.values()]

    def test_repeated_partitioned_and_empty(self):
        for circuit in self.circuits:
            D = circuit.compile_detector_sampler(seed=146200).sample(256)
            expected = self.decode(circuit, D)
            self.assertEqual(expected.dtype, np.uint8)
            np.testing.assert_array_equal(expected, self.decode(circuit, D))
            np.testing.assert_array_equal(expected, np.concatenate(
                [self.decode(circuit, D[:128]), self.decode(circuit, D[128:])]))
            np.testing.assert_array_equal(self.decode(circuit, D[:0]), np.empty((0, 4), np.uint8))

    def test_bad_input(self):
        circuit = self.circuits[0]
        for D in (np.zeros((1, 69)), np.full((1, 70), 2)):
            with self.assertRaises(ValueError):
                self.decode(circuit, D)
        with self.assertRaises(ValueError):
            self.decode(stim.Circuit(), np.zeros((1, 70)))

    def test_package_import_initializes_model(self):
        from mld_ensemble import decode
        from mld_ensemble.decoder import decode as implementation
        self.assertIs(decode, implementation)


if __name__ == "__main__":
    unittest.main()
