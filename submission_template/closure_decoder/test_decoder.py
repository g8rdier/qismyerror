"""Small, self-contained correctness tests; no training/test CSVs required."""
import csv
import hashlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest

import numba
import numpy as np
import stim

if not __package__:
    import decoder as _standalone_entry
from closure_decoder import bp_osd, candidates, closure, prepare

HERE = Path(__file__).resolve().parent


class TestAlgebra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(min(4, numba.get_num_threads()))

    def test_every_uint64_bit(self):
        for i in range(64):
            self.assertEqual(candidates.ctz(np.uint64(1 << i)), i)

    def test_exact_small_posterior(self):
        rng = np.random.default_rng(49102)
        settings = {"schedules": [[3, .8, .5], [5, .8, .5], [8, .8, .5]],
                    "lam": 2, "lam2": 2, "capacity": 64, "gap": 1000.,
                    "closure_seeds": 2}
        for _ in range(4):
            H = np.concatenate([np.eye(4, dtype=np.uint8),
                                rng.integers(0, 2, (4, 2), dtype=np.uint8)], axis=1)
            L = rng.integers(0, 2, (4, 6), dtype=np.uint8)
            p = rng.uniform(.01, .3, 6)
            graph = bp_osd.make_graph(H, L, p)
            D = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.uint8)
            _, cp = bp_osd.solve(graph, D, settings, return_probabilities=True)
            e = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(np.uint8)
            syndrome = e @ H.T % 2
            logical = (e @ L.T % 2) @ (1 << np.arange(4))
            mass = np.prod(np.where(e, p, 1 - p), axis=1)
            for i, d in enumerate(D):
                valid = (syndrome == d).all(1)
                exact = np.bincount(logical[valid], weights=mass[valid], minlength=16)
                exact /= exact.sum()
                np.testing.assert_allclose(cp[i], exact, atol=2e-6)

    def test_duplicate_checkpoints_do_not_multiply_mass(self):
        H = np.array([[1, 0, 1, 1], [0, 1, 1, 0]], np.uint8)
        graph = bp_osd.make_graph(H, np.eye(4, dtype=np.uint8), np.array([.1, .2, .3, .15]))
        D = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.uint8)
        settings = {"schedules": [[5, .8, .5]], "lam": 2, "lam2": 2,
                    "capacity": 64, "gap": 1000., "closure_seeds": 2}
        _, a = bp_osd.solve(graph, D, settings, return_probabilities=True)
        settings["schedules"] *= 3
        _, b = bp_osd.solve(graph, D, settings, return_probabilities=True)
        np.testing.assert_array_equal(a, b)

    def test_hash_collision_is_not_a_duplicate(self):
        start, mul, mask = 0xCBF29CE484222325, 0x100000001B3, (1 << 64) - 1
        ha, hb = start * mul & mask, (start ^ 1) * mul & mask
        a, b = np.array([0, 0], np.uint64), np.array([1, ha ^ hb], np.uint64)
        self.assertEqual(candidates.vector_hash(a), candidates.vector_hash(b))
        vectors = np.empty((8, 2), np.uint64)
        costs, labels, hashes = np.empty(8), np.empty(8, np.uint8), np.empty(8, np.uint64)
        index = candidates.create(8)
        used, _ = candidates.put(a, 1., np.uint8(1), vectors, costs, labels, hashes, 0, index)
        used, _ = candidates.put(b, 2., np.uint8(2), vectors, costs, labels, hashes, used, index)
        self.assertEqual(used, 2)
        used, _ = candidates.put(a, 1., np.uint8(1), vectors, costs, labels, hashes, used, index)
        self.assertEqual(used, 2)

    def test_heap_replacement_and_ties(self):
        rng = np.random.default_rng(2844)
        capacity = 8
        vectors = np.empty((capacity, 2), np.uint64)
        costs, labels, hashes = np.empty(capacity), np.empty(capacity, np.uint8), np.empty(capacity, np.uint64)
        index, used, expected = candidates.create(capacity), 0, []
        for i in range(500):
            vector = np.array([i + 1, i + 2], np.uint64)
            cost = float(rng.integers(-i // 20, 20))
            label = np.uint8(i % 16)
            if len(expected) < capacity:
                expected.append((vector.copy(), cost, label))
            else:
                worst = int(np.argmax([x[1] for x in expected]))
                if cost < expected[worst][1]:
                    expected[worst] = (vector.copy(), cost, label)
            used, _ = candidates.put(vector, cost, label, vectors, costs, labels, hashes, used, index)
            self.assertEqual(used, len(expected))
            for k, (v, c, lg) in enumerate(expected):
                np.testing.assert_array_equal(vectors[k], v)
                self.assertEqual(costs[k], c)
                self.assertEqual(labels[k], lg)

    def test_pair_index_both_words_and_collision_chains(self):
        rng = np.random.default_rng(3184)
        cols = np.zeros((16, 2), np.uint64)
        cols[:, 0] = rng.integers(0, 1 << 63, 16, dtype=np.uint64)
        cols[:, 1] = rng.integers(0, 64, 16, dtype=np.uint64)
        cols[14], cols[15] = cols[2], cols[3]
        heads, nxt, ii, jj = closure.build_index(cols)
        for a in range(16):
            for b in range(a + 1, 16):
                lo, hi = cols[a, 0] ^ cols[b, 0], cols[a, 1] ^ cols[b, 1]
                k = closure.find(lo, hi, cols, heads, ii, jj)
                got = set()
                while k >= 0:
                    got.add((int(ii[k]), int(jj[k])))
                    k = nxt[k]
                exact = {(i, j) for i in range(16) for j in range(i + 1, 16)
                         if (cols[i, 0] ^ cols[j, 0]) == lo and (cols[i, 1] ^ cols[j, 1]) == hi}
                self.assertEqual(got, exact)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = importlib.import_module("closure_decoder.decoder")

    def test_asset_identity(self):
        self.assertEqual(hashlib.sha256((HERE / "assets.npz").read_bytes()).hexdigest(),
                         self.model.MODEL["assets_sha256"])
        self.assertEqual(len(self.model._STATES), 2)

    def test_shape_determinism_duplicates_and_batch_size(self):
        for state in self.model._STATES.values():
            c = stim.Circuit(state["circuit"])
            D = c.compile_detector_sampler(seed=184012).sample(80)
            pred = self.model.decode(c, D)
            self.assertEqual(pred.shape, (80, 4))
            self.assertEqual(pred.dtype, np.uint8)
            self.assertTrue(np.isin(pred, (0, 1)).all())
            np.testing.assert_array_equal(pred, self.model.decode(c, D))
            np.testing.assert_array_equal(pred[::-1], self.model.decode(c, D[::-1]))
            np.testing.assert_array_equal(pred[:20], self.model.decode(c, D[:20]))
            np.testing.assert_array_equal(np.concatenate([pred, pred]),
                                          self.model.decode(c, np.concatenate([D, D])))
            self.assertEqual(self.model.decode(c, D[:0]).shape, (0, 4))

    def test_invalid_inputs_and_unknown_circuit(self):
        state = next(iter(self.model._STATES.values()))
        c = stim.Circuit(state["circuit"])
        with self.assertRaises(ValueError):
            self.model.decode(c, np.zeros((2, 69)))
        with self.assertRaises(ValueError):
            self.model.decode(c, np.full((2, 70), 2))
        with self.assertRaises(ValueError):
            self.model.decode(stim.Circuit(), np.zeros((2, 70)))


class TestSubmission(unittest.TestCase):
    def test_text_checksums_survive_git_line_endings(self):
        with tempfile.TemporaryDirectory(prefix="closure-text-") as directory:
            a, b = Path(directory) / "lf.csv", Path(directory) / "crlf.csv"
            a.write_bytes(b"id,L0\nshot,1\n")
            b.write_bytes(b"id,L0\r\nshot,1\r\n")
            self.assertNotEqual(prepare.sha256(a), prepare.sha256(b))
            self.assertEqual(prepare.text_sha256(a), prepare.text_sha256(b))

    def test_csv_schema_and_order(self):
        ids = np.array(["shot_b", "shot_a"])
        pred = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], np.uint8)
        payload = prepare._submission(ids, pred)
        rows = list(csv.reader(io.StringIO(payload.decode())))
        self.assertEqual(rows[0], ["id", "L0", "L1", "L2", "L3"])
        self.assertEqual(rows[1], ["shot_b", "1", "0", "1", "0"])
        self.assertEqual(rows[2], ["shot_a", "0", "1", "0", "1"])
        self.assertEqual(payload, prepare._submission(ids, pred))
        with tempfile.TemporaryDirectory(prefix="closure-test-") as directory:
            path = Path(directory) / "submission.csv"
            path.write_bytes(payload)
            prepare._check_csv(path, {"id": ids})
            with self.assertRaises(ValueError):
                prepare._check_csv(path, {"id": ids[::-1]})

    def test_nonbinary_submission_rejected(self):
        with self.assertRaises(ValueError):
            prepare._submission(["shot"], np.array([[0, 2, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
