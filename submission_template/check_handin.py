# check_handin.py: run inside the hand-in folder, in the fresh environment (the organisers' script from the Kaggle page)
import time
import numpy as np
import stim
from decoder import decode

total = 0.0
for basis in "XZ":
    circuit = stim.Circuit.from_file(f"{basis}_circuit.stim")
    events, flips = circuit.compile_detector_sampler(seed=1).sample(100_000, separate_observables=True)
    start = time.perf_counter()
    predictions = np.asarray(decode(circuit, events))
    total += time.perf_counter() - start
    assert predictions.shape == flips.shape and np.isin(predictions, (0, 1)).all()
    print(f"{basis}: LER {np.mean(predictions != flips):.5f}")
print(f"decode time: {total:.1f} s")
