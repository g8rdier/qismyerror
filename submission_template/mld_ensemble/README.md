# Ready-to-use MLD ensemble

Everything for this decoder is in this folder. **No training is needed before
use.** Runtime dependencies are only NumPy, Stim and Numba; it does not import
the parent decoders, experimental folders, or `hackathon_2026`.

## Data and accuracy

This is an **analytic physics decoder**, not a fitted neural network:
**0 fitted training shots and no neural weights**. The frozen `assets.npz`
(22 KB) contains both circuits, their H/L matrices, physical error probabilities
and detector permutations. Generating extra shots does not fit new parameters.

- Settings selected using 66,000 development shots.
- Earlier independent evaluation: **500,000 shots, LER 0.033548**.
- New release evaluation: **2,000,000 newly generated shots** (1M/basis,
  seeds 1086100/1086101): **LER 0.03364375**, shot-level 95% interval
  **0.03345966–0.03382784**. Detailed results: `validation.json`.
- Supplied labelled `data/train` reference: **LER 0.033860** on 200,000 shots.
- No test labels or leaderboard feedback are used.

This folder is about **4.1 MB**, mostly the requested 4 MB prediction CSV.
All eight tests pass; exported predictions match the research model exactly.

## How it works

This decoder is a **physics and probability calculation**, not a neural network
trained on labels. For each shot it asks: *which combinations of physical
faults could have produced these detector events, and how much probability
belongs to each possible logical answer?* The exact sum is too large to
enumerate, so the decoder computes a carefully pruned approximation.

The explanation below starts with the problem and builds up to the two-sweep
ensemble. The short version is:

1. Turn the circuit into a list of possible faults and their probabilities.
2. Sweep through those faults, accumulating all explanations consistent with
   the observed detectors.
3. Merge explanations that have become indistinguishable, and throw away the
   least promising ones.
4. Repeat the sweep in the opposite direction and in a different detector order.
5. For each of the 16 logical outcomes, keep the best estimate; convert those
   estimates into four per-bit probabilities and threshold at 0.5.

### 1. The problem in plain language

A shot produces a **syndrome**: 70 detector bits, `D0` to `D69`. A detector
fires when a parity check disagrees with its previous value. The syndrome is
evidence that some physical faults happened, but it does not say which ones.

The noisy circuit defines about **2,470 possible fault mechanisms**. Each
mechanism has a prior probability `p_i`. A mechanism can flip several detectors
and up to four **logical bits**, which record whether the stored logical state
was flipped. Mechanisms with exactly the same detector and logical effect are
merged into one column, so every column has a unique effect. The circuit gives
us two binary matrices:

| Symbol | Meaning |
|---|---|
| `D` | the observed detector events, 70 bits |
| `e` | the hidden error vector: one bit per mechanism, 1 means it fired |
| `p_i` | the prior probability that mechanism `i` fired |
| `H` | which detectors each mechanism flips |
| `L` | which logical bits each mechanism flips |
| `O` | the four logical flips we must predict |

In equations, the circuit tells us `D = H e (mod 2)` and `O = L e (mod 2)`.
The decoder knows `H`, `L` and `p`; it must infer the likely `O` from `D`.

The target is not “find the single most likely error vector.” The map from
faults to detectors is **many-to-one**: many different physical fault patterns
produce exactly the same `D`, and they can produce different logical flips.
Those alternatives must be weighted and added. The competition metric counts
each logical bit separately, so the right target is the **marginal probability**

```text
P(O_j = 1 | D)
```

for each bit `j`. Predicting 1 when that probability is above 0.5 minimizes the
expected number of wrong bits. This distinction—summing over explanations
rather than committing to one—is the central idea of marginal
(Bayes-optimal) decoding for this degenerate code.

### 2. Why the decoder needs a search

There are two enormous numbers in the problem:

- Each of the ~2,470 mechanisms can be on or off, so the error vector has about
  `2^2470` possible values.
- Even after grouping by the 70-bit syndrome, the number of distinct partial
  explanations that survive during a sweep is far too large to enumerate
  exactly on a CPU.

The decoder therefore uses a **dynamic programme** to build explanations one
mechanism at a time, plus a **beam search** to discard weak partial explanations
early. “Dynamic programme” here simply means: solve the problem in stages and
reuse the partial answers from the previous stage. “Beam” means: keep a bounded
set of the most promising partial answers instead of all of them.

### 3. What one “state” contains

During the sweep, the decoder does not remember every fault that has fired. Two
different partial fault patterns are equivalent if they have:

1. the same **residual syndrome**—the detector bits that still disagree with
   the observed `D`; and
2. the same **logical flips**—the four bits produced by the faults chosen so far.

Future mechanisms only care about those two things. For example, if two
different faults both leave detector 7 unresolved and neither has flipped a
logical bit, the rest of the calculation cannot tell them apart. Their
probabilities can be added immediately. This is why a state stores only:

```text
residual syndrome (70 bits) + logical flips (4 bits)
```

plus one number: the total probability mass of all partial fault patterns that
have reached this summary. “Mass” is measured relative to the empty error
vector: the common factors `(1 - p_i)` cancel, so a mechanism that fires
contributes a factor `p_i / (1 - p_i)`, while one that does not fire contributes
1.

The implementation packs the 70 residual bits and 4 logical bits into **two
64-bit words**. That compact representation is what makes a hash table of
states practical.

### 4. The sweep: one mechanism at a time

The decoder sorts mechanisms by the **first detector they touch**—ties are
broken by the last detector. This ordering is important: once the last
mechanism touching a detector has been processed, that detector can never be
changed again. If a state still disagrees with `D` on it at that point, the
state is impossible and is dropped immediately.

For each mechanism `k`, every surviving state has two possible children:

- **Did not fire.** The residual and logical bits stay the same, and the
  probability mass is unchanged.
- **Fired.** The mechanism’s detector column is XORed into the residual, its
  logical column is XORed into the logical bits, and the mass is multiplied by
  `p_k / (1 - p_k)`.

The “did not fire” children are updated in place, because they are just the
same state with a slightly different search score. The “fired” children are
looked up in a hash table: if the same residual-and-logical summary already
exists, the two masses are added; otherwise a new state is created.

At the end, the decoder collects only states whose residual is zero. They still
differ in their four logical bits, so their masses are grouped into **16
logical classes** (`0000` through `1111`). Those 16 masses are the decoder’s
approximation to the posterior over logical outcomes.

```text
Start: residual = D, logical = 0000, mass = 1
  for each mechanism:
      keep a copy (mechanism did not fire)
      or XOR its detector effect into the residual
         and its logical effect into the logical bits,
         multiplying the mass by p/(1-p)
      merge equal summaries
      close detectors whose last mechanism has passed
Finish: residual = 000000...0
        → 16 logical masses
```

### 5. Beam pruning: deciding what to forget

The exact state table is still too large. The decoder gives every state a
**search score** made of two parts:

```text
score = log(probability mass) + look-ahead estimate
```

The first part rewards explanations that already have high probability. The
second part is needed because an explanation that has not yet fixed a fired
detector looks incomplete; without a look-ahead, the “no faults at all” state
can look best until it is too late.

For each detector that a state still disagrees with, the look-ahead charges an
estimate of the cost of fixing it. The estimate uses the most likely future
mechanism that touches that detector, multiplied by `heuristic_exponent = 0.5`.
The factor 0.5 is a heuristic: a typical fault touches about two detectors, so
roughly half of its “cost” is assigned to each detector. This is only a
search-ordering heuristic; it does **not** change the physical probability mass
of a completed explanation.

After each step, the decoder:

- drops any state whose score is below `best + log(eps)`—that is, more than
  `-log(eps)` below the best score;
- if there are more than `max_states`, keeps only the highest-scoring ones; and
- periodically compacts the table (every 32 steps, and whenever a detector
  closes).

The two passes use different settings:

| Pass | Direction | `eps` | Max states | Runs on |
|---|---|---:|---:|---|
| First | forward | `1e-4` | 1,024 | every shot |
| Refinement | backward | `3e-5` | 4,096 | uncertain shots only |

A smaller `eps` and a larger state cap keep more explanations and cost more
time. The refinement pass runs only when any logical bit is still uncertain:
specifically, when a bit’s current probability lies between **0.15 and 0.85**.
Easy shots stop after the first pass.

### 6. Why there are forward, backward and shuffled sweeps

Pruning is not exact. A forward sweep may discard a low-probability partial
explanation that would have become important later. A backward sweep builds the
same kind of dynamic programme with the detector order reversed, so it makes
its pruning decisions from the other end and may retain a different set of
explanations.

Each sweep returns **absolute log masses** for the 16 logical classes: masses
relative to the all-no-error explanation, on the same scale for every sweep of
that shot. Pruning only removes positive terms from a sum, so each sweep’s mass
is a **lower bound** on the exact mass for that class. The decoder combines two
sweeps by taking the **maximum mass per class**:

```text
combined_mass[class] = max(mass_from_sweep_1[class],
                            mass_from_sweep_2[class])
```

Taking the maximum keeps the better of two lower bounds. It is not an average,
and it deliberately does not add the two sweeps together: adding would
double-count explanations that both sweeps found. This is still an
approximation, not an exact posterior.

The ensemble repeats this whole process with a second check order: component 0
uses the original chronological order, and component 1 uses a fixed permutation
**within each time block** (initial round, each middle round, and final
measurement). The permutation keeps the time structure but changes which
detector is “first” inside a round. That changes the mechanism order and
therefore which partial explanations survive pruning. The two component results
are again combined by a per-class maximum.

There is no randomness at decode time: the shuffled order is stored in
`assets.npz`, and all search settings are fixed in `model.json`.

### 7. From 16 masses to four predictions

The 16 classes correspond to the 16 possible combinations of four logical bits.
For each bit `j`, the decoder sums the masses of all classes whose bit `j` is 1:

```text
P(L_j = 1 | D) = sum of masses of classes with bit j set
```

It then predicts 1 when that probability is greater than 0.5. These are
**per-bit marginal decisions**, so the four predicted bits need not all come
from one single logical class. That is intentional: the competition scores each
bit independently, and the per-bit marginal is the decision that minimizes the
expected number of wrong bits.

If no complete explanation survives at all, every class is tied at probability
0.5; the strict `> 0.5` threshold then produces a deterministic all-zero
prediction.

### 8. What the code does at runtime

The heavy preparation happens when the module is imported, before the timed
`decode()` calls:

- `assets.npz` stores the two circuit texts, their `H`, `L` and `p` values, and
  the two detector permutations.
- `build_mld` precomputes the mechanism order, column bitmasks, detector-closing
  times and all look-ahead tables for every component and direction.
- The Numba kernel compiles and is warmed on a zero-syndrome shot.

A `decode()` call then:

1. fingerprints the circuit and selects its frozen tables;
2. deduplicates exact repeated detector patterns within the call;
3. runs the two components, each with its forward pass and optional backward
   refinement;
4. takes the per-class maximum and thresholds the four marginals; and
5. maps predictions back to the original shot order.

Shots are parallelized with Numba. Duplicate patterns are decoded once inside a
call; there is no memoization between calls and no reading of `results.csv`.
The model is deterministic and produces identical predictions across repeated
calls.

### 9. What “training” means here

Nothing in `assets.npz` is a learned neural weight. The probabilities come from
the circuit’s SI1000 noise model (`p = 0.002`), and the rest is deterministic
linear algebra and search. The `fitted_training_shots` count in `model.json` is
therefore **0**.

The 66,000 development shots and larger validation sets were used to choose and
measure settings such as beam caps, refinement thresholds and detector orders,
not to fit a model that stores per-example answers. Generating more labelled
shots can give a more precise LER estimate, but it does not change the decoder’s
parameters or make the search itself more accurate. The supplied `data/test/test.csv` labels are
not available and are never used.

A useful one-sentence summary: **the decoder repeatedly adds up the probability
of every plausible fault pattern that could explain the observed detectors, uses
beam pruning and repeated sweep orders to keep that sum tractable, and predicts
each logical bit from the resulting marginal probability.**

The model initializes at import. Calls are deterministic and CPU-parallel;
48 judge threads are supported. Prior local timing was 48.98s / 200k shots on
16 laptop threads, **not** a B200/48-core timing.

For code readers: `kernel.py` contains the state sweep and pruning;
`decoder.py` contains the forward/backward routing, two-order ensemble and
bit-marginal decision; `assets.npz` contains the frozen circuit physics and
orders; and `prepare.py` rebuilds those assets or runs evaluation/prediction.

## Runtime improvement ideas (not implemented)

The measured local baseline is **48.98s per 200k shots** (median of five,
16 laptop threads). At perfect 3x thread scaling that would be about 16.3s on
48 threads, close to the 15.4s reference cap; real B200/48-core timing is
unknown. Since the ensemble gains only **0.14% relative LER**, reducing runtime
has unusually high leverage on the final score. The ideas below are sketches,
not measured wins. Each needs paired accuracy and five-run timing on fresh data.

### 1. Do not run both components on every shot

This is the first thing to try. Run the chronological component for all shots,
then run the shuffled component only where the first result is uncertain.
Thresholds can be calibrated by replaying the 500k development/audit predictions
(the cache lives in the research branch) or by regenerating equivalent samples;
no new labels are needed. The second component is roughly half the work,
so routing at a 10–20% uncertainty rate could remove 35–45% of runtime. The
cost is part of the ensemble's small accuracy gain, which must be measured,
not assumed.

Also test a **single-component mode** using the already-developed
hybrid/tighter search: it measured **31.47s** and LER **0.033560** on the 500k
comparison, versus 48.98s and **0.033548** for the ensemble. That is a 36%
runtime cut for 24 extra wrong bits out of 2M. Once runtime is scored, this may
be the better operating point. A `combination: "single"` switch in `model.json`
would make it a small, reversible change.

### 2. Add a safe head fast path

About **4.6%** of shots have no fired detectors and **28%** have at most four
(4.6% zero plus 23% one-to-four). Precompute 16 logical-outcome weights offline
with the full kernel for the covered syndromes that fit a chosen table budget,
store them in `assets.npz`, and use them only when the margin is safely outside
the table's truncation error. Fall back to the full dynamic programme
otherwise. Precompute probabilities, **not majority-vote labels**, so the
per-bit marginal decision is preserved. Build the table from physical faults or
development seeds only—never from test labels—and validate on untouched seeds.

### 3. Stop a shot as soon as its answer is fixed

Add a suffix logical-influence mask: for each sweep step, know which logical
bits any remaining mechanism can still flip. If all surviving states agree on a
bit and no remaining column can flip it, that bit is final. When all four bits
are final, emit the result and skip the rest of the shot. A weaker variant is
to stop when all surviving states share one full 4-bit class and the remaining
suffix cannot introduce another class. Both are exact for the pruned beam if
the mask is conservative. Require bit-identical predictions against the current
kernel on the full 500k audit set before accepting them.

### 4. Reduce kernel and cache overhead

Candidate, accuracy-preserving micro-optimizations:

- Store hash-table indices as `uint16`/`uint32` instead of `int64`. At
  `max_states <= 4096` the table fits comfortably and the smaller footprint
  should reduce cache misses.
- Rebuild the hash table with generation stamps instead of clearing and
  reinserting every state at each compaction.
- Replace the full top-k `argsort` with `partition` when the state cap is hit.
- Specialize the compiled kernel for the fixed `70` detectors and `4`
  observables instead of passing them as runtime arguments.
- Pack `D0/D1` vectorized in NumPy once per batch instead of looping over 70
  detectors inside every shot.
- Test `float32` state weights/scores. This is the riskiest item: it can change
  rare threshold decisions, so it requires exact-prediction and full-audit
  checks, not just a lower timing number.

Validate each change independently. A 5–15% kernel improvement is useful, but
not at the cost of even a few extra wrong logical bits.

### 5. Improve 48-thread load balance

The kernel uses static contiguous `prange` ranges. If hard shots cluster, a few
threads finish last while the rest idle. Flatten `(component, shot)` into one
parallel batch, especially when routing leaves few shots in the second stage,
and assign work in a cost-aware interleaved order (for example, round-robin by
fired-detector count). Do **not** simply sort all easy and hard shots together:
the earlier sorted-deduplication experiment was about 19% slower. Preserve
random input order when in doubt. Re-time on 200k shots, and ideally on a
48-thread machine.

### 6. Revisit the ensemble before optimizing it

Compare LER-versus-runtime points on the same held-out set:

- original reference MLD: 21.53s (one diagnostic run), LER 0.033595;
- hybrid/tighter single component: 31.47s, LER 0.033560;
- two-component ensemble: 48.98s, LER 0.033548.

The ensemble buys **0.000012 LER** over the hybrid—24 fewer wrong bits out of
2M—while costing 56% more. A slightly wider **single** beam may dominate both.
Optimize the scored objective `LER * max(1, t / t_cap)^alpha`, not LER alone.

### 7. GPU is a later, larger project

The dynamic programme is branchy, hash-table heavy and serial within a shot, so
it is not a drop-in GPU port. More plausible uses are a GPU-scored head table,
or a batched dense beam used only for hard shots. The B200 has enormous compute,
but a poor GPU mapping can easily lose to 48 CPU threads. Treat this as a
research track after the CPU routing and kernel work, not as an expected quick
win.

### Measurement protocol

Use new development and untouched audit seed families, paired shot-level
confidence intervals, and median-of-five `run_local.py` timings. Keep
predictions deterministic and bit-identical across runs. Import-time
precomputation is free in the competition, but it must not read the test labels,
`results.csv`, or memoize test predictions. Report the accuracy/runtime Pareto
frontier rather than a single cherry-picked setting.

## Use / swap models

```powershell
.venv/Scripts/python.exe submission_template/run_local.py --decoder submission_template/mld_ensemble --problem-set data/train --runs 5 --threads 16 --out check.csv
```

To select it as the main decoder, replace `submission_template/decoder.py`
with the included `entrypoint.py`, or simply use this module-level import:

```python
from mld_ensemble.decoder import decode
```

Equivalently: `Copy-Item submission_template/mld_ensemble/entrypoint.py submission_template/decoder.py`.

The main entry point is deliberately **not changed** by this release, so
colleagues can replace this one folder or choose another model independently.

`results.csv` is the prediction for `data/test/test.csv`, with header
`id,L0,L1,L2,L3`, original input order and binary values. **Inference never reads
this CSV**, validation reports or saved predictions.
`model.json` records the input/output hashes; `manifest.json` fingerprints the
release files. Input CSV hashes tolerate LF/CRLF checkout differences.

## Reproduce

Use a **new** data output directory; `generate_data.py` wipes existing files.

```powershell
.venv/Scripts/python.exe data/generate_data.py data/mld_generated --shots 1000000 --seed 1086100
.venv/Scripts/python.exe submission_template/mld_ensemble/prepare.py evaluate --data data/mld_generated/mld_generated.csv --out NEW_VALIDATION.json
.venv/Scripts/python.exe submission_template/mld_ensemble/prepare.py predict --data data/test/test.csv --out NEW_RESULTS.csv
.venv/Scripts/python.exe submission_template/mld_ensemble/prepare.py verify --data data/test/test.csv --out submission_template/mld_ensemble/results.csv
.venv/Scripts/python.exe submission_template/mld_ensemble/prepare.py build --out NEW_ASSETS.npz
.venv/Scripts/python.exe -m unittest discover -s submission_template/mld_ensemble -v
```

`build` can also take `--circuits data/train`. It rebuilds the small physics
asset, not a fitted model. Tools refuse to overwrite existing outputs. Pin the
versions in `requirements.txt` when reproducing samples and assets.
For a new release, rerun validation and prediction, update `model.json`, then
run `prepare.py manifest` in a fresh release folder. Do not mix versioned assets.
