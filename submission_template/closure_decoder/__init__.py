"""Optional pretrained decoder with one canonical namespace for Numba caches."""
from pathlib import Path
import sys

# ``submission_template.closure_decoder`` and a standalone ``import decoder``
# must resolve compiled helpers under the same name, never main's bp_osd.py.
_existing = sys.modules.setdefault("closure_decoder", sys.modules[__name__])
if Path(_existing.__file__).resolve() != Path(__file__).resolve():
    raise ImportError("a different closure_decoder is loaded; use a fresh process")


def __getattr__(name):
    if name == "decode":
        from closure_decoder.decoder import decode
        return decode
    raise AttributeError(name)
