# Optional pretrained closure decoder

This folder is self-contained. **It does not replace or modify
`submission_template/decoder.py`.** No training is needed to use `decode()`.
`assets.npz` contains both bases' trained lookup tables and public circuit
physics; `model.json` identifies the exact configuration and validation.

**Release validation:** LER **0.034319** on **2,000,000 new simulated shots**
(1M per basis; 95% shot-level interval **0.034132–0.034505**).
No retuning used this audit. The lookup was trained on 10M shots per basis;
a fresh 50M/basis challenger showed no convincing development improvement.
The ready-built asset is **540 KB** and its generator reproduced it byte-for-byte.

## Use it

From the repository root:

```powershell
uv pip install -r submission_template/closure_decoder/requirements.txt
.venv/Scripts/python.exe submission_template/run_local.py --decoder submission_template/closure_decoder --problem-set data/train --threads 16 --runs 5 --out closure_train_check.csv
```

Normal Python imports also work:

```python
from submission_template.closure_decoder import decode
predictions = decode(circuit, detector_events)  # (shots, 70) -> uint8 (shots, 4)
```

To select this model in the existing main entry point later, add a
**module-level** import and delegate its `decode()` body:

```python
from closure_decoder import decode as decode_closure  # before the timing starts

def decode(circuit, detector_events):
    return decode_closure(circuit, detector_events)
```

That optional edit is deliberately not made by this release.
For another model, replace this folder or delegate to another implementation
with the same interface.

Tested with Python 3.13 and the pinned dependencies below. Git attributes keep
release files at LF; text checksums tolerate LF/CRLF input CSV checkouts.

## How it works

1. An exact lookup answers familiar easy detector patterns (at most four fired).
2. Damped belief propagation ranks likely physical faults at iterations 3/5/8.
3. GF(2) ordered-statistics search constructs explanations matching the syndrome.
4. A collision-safe pool combines distinct explanations without double-counting.
5. Physical-pair closure adds equivalent fault pairs, then each output bit is
   selected by its total candidate probability.

This is a physics-based decoder, not an MLP. The first import loads assets,
builds the pair index and compiles Numba; this initialization is not training.
Inference is CPU-parallel and honors 48 judge threads. GPU/judge-server timing
is not claimed; 15.4 seconds on a laptop is not a release cutoff.

## Predictions and provenance

`results.csv` is the model's output on the supplied unlabelled test CSV:
`id,L0,L1,L2,L3`, in original row order. **The decoder never reads this file.**
`model.json` records the input, source, asset and output hashes and the
independent validation results. No test labels are available or used.

```powershell
.venv/Scripts/python.exe submission_template/closure_decoder/prepare.py --threads 16 predict --data data/test/test.csv
.venv/Scripts/python.exe submission_template/closure_decoder/prepare.py verify --data data/test/test.csv
.venv/Scripts/python.exe -m unittest discover -s submission_template/closure_decoder -v
```

## Optional rebuilding

Use the shot count, seeds and chunk size recorded in `model.json`:

```powershell
.venv/Scripts/python.exe submission_template/closure_decoder/prepare.py train --circuits data/train --out NEW_TRAINING_DIR --shots 10000000 --chunk 2000000 --seed 20261113
.venv/Scripts/python.exe submission_template/closure_decoder/prepare.py build --circuits data/train --tables NEW_TRAINING_DIR --out NEW_ASSETS.npz
```

These commands never run during inference and refuse to overwrite existing
outputs. Stim regeneration depends on its pinned version and machine
architecture; the included assets are the frozen, ready-to-use model.
Do not substitute rebuilt assets without updating their checksum and validating.

Runtime code is split into `decoder.py` (API/assets), `bp_osd.py` (search),
`closure.py` (pair index/expansion), and `candidates.py` (exact pooling).
The folder contains no alternate model presets, training samples, or research
caches. Only numpy, Stim and Numba are required.
