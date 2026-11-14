"""Self-contained ensemble; importing ``decode`` also initializes its tables."""
__all__ = ["decode"]


def __getattr__(name):
    if name == "decode":
        from .decoder import decode
        globals()["decode"] = decode
        return decode
    raise AttributeError(name)
