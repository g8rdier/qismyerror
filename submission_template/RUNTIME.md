# Runtime optimization with frozen predictions

The selected submission is `submission_tweak_tieband01.csv`, SHA-256
`878631d6b0a31f7c2ec06d55b58fb82b7f6b8aa26f27e7321f5ebb63b08a17a1`.
Every retained change must reproduce every shot and all four logical bits.
Matching aggregate LER is insufficient.

The optimization in `mld.py` extends an existing condition from the small-state
loop to the vectorized loop. When `hasv0[k]` is false, every `cv0` coefficient
for that mechanism is exactly zero. The vectorized loop leaves the initialized
zero `ce` values alone instead of repeatedly multiplying and adding zeros.
Updates to `cs` retain their original operations and detector order. When a
nonzero `cv0` exists, the original arithmetic path runs unchanged.

`decoder.py`, physical probabilities, all three stages, pruning thresholds,
state caps, compaction frequency, refinement decisions, precision, and the
`TIE_BAND=0.01` output rule are unchanged. The scheduling, smaller-integer,
and native-bit-scan trial was discarded because its measured speed difference
was within observed variation. It is not part of this change.

## Reproduce the selected-submission check

Install the **hand-in** requirements from `submission_template/requirements.txt`.
The repository-root requirements are for development and include an editable
checkout of this repository; they are not needed by the standalone decoder.

```bash
python3.13 -m venv /tmp/qec-verify
/tmp/qec-verify/bin/python -m pip install -r submission_template/requirements.txt
/tmp/qec-verify/bin/python -I submission_template/verify_runtime.py \
  --data data/test/test.csv \
  --submission submission_tweak_tieband01.csv \
  --threads 48 --runs 5 --report /tmp/runtime-check.json
```

The verifier imports before timing, calls `decode` once per basis per run,
checks every output bit on every run, and additionally requires a byte-identical
CSV in the original input order. It reports the total X+Z runtime and source
hashes. It uses only the hand-in dependencies and does not import the research
project package or use labels. Local timings are not official-machine timings.

For intermediate numerical comparisons, run `verify_runtime_parity.py` once
per decoder revision in separate processes, with the same `--threads` setting:

```bash
/tmp/qec-verify/bin/python -I submission_template/verify_runtime_parity.py \
  --decoder /path/to/reference-folder --threads 16 --out /tmp/reference.npz
/tmp/qec-verify/bin/python -I submission_template/verify_runtime_parity.py \
  --decoder submission_template --threads 16 --out /tmp/candidate.npz
```

Both folders need `decoder.py`, `mld.py`, and `circuits/{X,Z}_circuit.stim`.
The script checks 2,048 fresh shots per basis, reversed and partitioned batches,
duplicates, empty input, exact bit-index operations, and the zero-coefficient
condition. It saves all intermediate stage log weights, final marginals, and
predictions. Compare the NPZ key sets and every corresponding array using
`numpy.testing.assert_array_equal`; equality tolerances must not be used.

The judge's minimal decoder folder still needs only `decoder.py`, `mld.py`,
`requirements.txt`, and `circuits/{X,Z}_circuit.stim`. The verification tools
and reports are for reproducibility and are never loaded by `decode`.

## First optimization: local validation

On an AMD Ryzen 7 250 (8 cores, 16 hardware threads), both bases and all
200,000 test shots, with import/JIT excluded:

| Implementation | Threads | Runs | Median X+Z runtime | Changed prediction bits |
| --- | ---: | ---: | ---: | ---: |
| Original | 16 | 3 | 88.94 s | 0 |
| Skip zero updates | 16 | 3 | 86.10 s | 0 |
| Skip zero updates | 48 | 1 | 86.13 s | 0 |

The 16-thread median improved by 3.2%. Rechecking the original afterward
took 89.63 s. Every full run reproduced all 800,000 prediction bits and the
entire selected CSV byte for byte. All 21 generated-data arrays, including
every stage's log weights and final marginals, matched the original exactly
with 1 and 16 candidate threads.

These are local measurements, not the organisers' median of five on their
48-vCPU machine. Full timings, package versions, source hashes, and static
checks are recorded in `runtime_validation.json`.

## Faster state ordering

The next change replaces repeated full floating-point argsorts with an exact
prefix sorter and gathers the retained state fields in one pass. It preserves
the original Numba quicksort pivots, swaps, insertion-sort cutoff, and tie
ordering. Order-preserving unsigned keys represent the existing float64 scores;
NaNs and signed zeros retain their original comparison equivalence. All
candidates are still considered. Only subpartitions wholly beyond the retained
prefix are left unsorted, so retained states and their order are unchanged.

This sorter adapts Numba's general-purpose quicksort, with its BSD copyright
and license notice included in `mld.py`. No external QEC implementation is used.
Decoder settings, numerical precision, probability calculations and dependencies
remain unchanged.

The local 16-thread median over three full runs is **78.34 s**, compared with
**86.10 s** for the previously pushed zero-update optimization and **88.94 s**
for the original. That is another **9.0%** improvement, or **11.9%** overall.
Every run matches all 800,000 selected-submission bits and the CSV bytes exactly.

Run the independent sorting tests with the same hand-in environment:

```bash
/tmp/qec-verify/bin/python -I submission_template/test_runtime_invariants.py
```

The tests exhaust small tied-value arrays and all binary arrays at the first
quicksort partition size, then check larger random, sorted, reversed, infinite,
NaN, signed-zero, subnormal and raw float-bit inputs. They require exact index
ordering and verify that the input score arrays remain bit-identical.

The final full 48-thread local check took **77.01 s**, with zero changed bits.
Fresh generated-shot intermediate arrays also match the original exactly at
1 and 16 threads. The compaction-copy trial was discarded because it showed
no reliable additional benefit. Detailed evidence is in the
`state_sort_optimization` section of `runtime_validation.json`.

## Hand-in folder

`handin/` (repository root) is the folder to hand in. It contains only what
`decode` needs: `decoder.py`, `mld.py`, `requirements.txt` (the three pinned
packages) and `circuits/{X,Z}_circuit.stim`. The Python files and circuits are
byte-identical copies of the ones in `submission_template/`; `decode` loads no
other file and imports none of the verification tools.

- `verify_runtime.py --decoder handin --threads 48`: all 800,000 bits and the
  CSV bytes match the selected submission.
- `check_handin.py` in a fresh Python 3.13.15 venv built from
  `handin/requirements.txt`: passes (X LER 0.03430, Z LER 0.03277).

## In-place hash-table renumbering

After every compaction the kernel used to clear the hash table and insert every
surviving state again (random memory accesses, ~22% of stage-3 kernel time).
The states keep their keys, so every occupied slot stays on its key's probe
sequence. The table is now renumbered in one sequential pass: surviving states
get their new index (also after the state-cap sort), removed states become a
marker (-2) that lookups skip and inserts may reuse. A full rebuild still
happens when occupied slots including markers exceed half the table, or when
the table is more than 16 times the state count. Hash keys, scores, weights,
state order and all arithmetic are unchanged.

| Implementation | Threads | Runs (interleaved) | Median X+Z runtime | Changed bits |
| --- | ---: | ---: | ---: | ---: |
| State-sort optimization | 16 | 3 | 77.33 s | 0 |
| + in-place renumbering | 16 | 3 | **74.52 s** | 0 |
| + in-place renumbering | 48 | 1 | 80.16 s | 0 |

Ranges: 73.05-75.14 s vs 77.01-77.83 s. Single-thread kernel time on stage-3
shots fell 14% and on stage-2 shots 11%; stage 1 is unchanged. All 21 parity
arrays are byte-identical to the reference at 1 and 16 threads, the invariant
tests pass, and `handin/` (updated to the same `mld.py`) reproduces the
selected CSV byte for byte and passes `check_handin.py` in a fresh venv.

## Trials after the state-sort optimization (not kept)

Profile at 16 threads, per basis: stage 1 ~3.9 s, stage 2 ~8 s, stage 3 ~26 s,
serial NumPy glue 0.18 s. Within stage 3 the exact prefix quicksort takes
~54% of the kernel time and the hash-table rebuild after compaction ~22%.

| Trial | Exact | Result |
| --- | --- | --- |
| Hardest-first scheduling of stages 2-3 (previous-stage peak states) | yes | within noise locally; later included (see below) |
| Radix sort, quicksort fallback on ties in the kept prefix | yes | slower: ties occur in nearly every sort |
| Quicksort into preallocated scratch buffers | yes | no gain |
| Skip the table rebuild when compaction removes nothing | yes | no gain |
| Swap state arrays after the cap sort instead of copying back (on top of in-place renumbering) | yes | no gain (stage 3: 15.17/15.51 s vs 15.33/15.40 s) |
| Decode duplicate syndromes once | — | duplicates are almost all cheap stage-1 shots; not implemented |

The quicksort's order of tied scores determines later tie-breaks and the
summation order of the final class weights, so any faster sort must reproduce
it exactly; no such sort was found. Details are in `runtime_validation.json`.

## Hardest-first scheduling (included on simulation evidence)

Stages 2 and 3 now hand out their shots most expensive first, using each
shot's peak state count from the previous stage (rank correlation 0.96 with
its time), in batches of about equal estimated cost. The results are put back
in the original shot order before any further arithmetic, and `decoder.py`
only passes the costs along; the escalation mask is computed as before.

A simulation from measured per-shot times predicts a shorter stage-3 tail on
48 cores (9.11 s to the optimal 8.34 s per basis). On the 8-core laptop the
gain is within noise (full set, 16 threads: 76.27/75.89 s before vs
76.70/75.61 s after; one-third subset, 48 threads: medians 26.31 s vs
26.12 s), so it was included at the team's request, not on measured speedup.
All 800,000 bits and the CSV bytes match at 16 and 48 threads, all 21 parity
arrays are byte-identical at 1 and 16 threads, and `check_handin.py` passes in
a fresh venv.

## Partition scans without bounds tests

In the exact prefix quicksort, the partition scans `values[i] < pivot` and
`pivot < values[j]` no longer test `i < high` and `j >= low`. After the
median-of-three step `values[high]` holds the pivot (never swapped inside the
loop) and `values[low]` stays <= pivot (it is only ever replaced by a value
<= pivot), so the scans stop exactly where they did. Pivots, swaps and the
returned order are unchanged; the sort invariant tests pass.

| Implementation | Threads | Runs (interleaved) | X+Z runtime | Changed bits |
| --- | ---: | ---: | ---: | ---: |
| Before (with hardest-first scheduling) | 16 | 2 | 73.76 s, 73.63 s | 0 |
| Partition scans without bounds tests | 16 | 2 | **64.24 s, 65.00 s** | 0 |
| Partition scans without bounds tests | 48 | 1 | 68.45 s | 0 |

All 21 parity arrays are byte-identical at 1 and 16 threads, and `handin/`
reproduces the selected CSV byte for byte and passes `check_handin.py` in a
fresh venv. The same idea for the insertion-sort step and int32 sort indices
gave no gain and were not kept.
