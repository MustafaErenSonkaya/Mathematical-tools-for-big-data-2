"""Sequential change detectors with one common interface (see base.py)."""
from .base import Detector
from .cusum import CUSUM
from .ewma import EWMA, ewma_limit
from .mewma import MEWMA
from .shewhart import Shewhart
from .sprt import SPRT

__all__ = ["Detector", "CUSUM", "EWMA", "MEWMA", "SPRT", "Shewhart", "ewma_limit", "METHODS",
           "THRESHOLD_PARAM", "SHAPE_PARAM", "make"]

METHODS = {"shewhart": Shewhart, "sprt": SPRT, "cusum": CUSUM, "ewma": EWMA, "mewma": MEWMA}

# the parameter that sets the alarm limit; calibrate.py tunes it to a target ARL0
THRESHOLD_PARAM = {"shewhart": "k", "sprt": "alpha", "cusum": "h", "ewma": "L", "mewma": "h"}
# the parameter that sets the "shape" (which shift the method is tuned for)
SHAPE_PARAM = {"shewhart": None, "sprt": "delta", "cusum": "kappa", "ewma": "lam", "mewma": "lam"}


def make(method: str, **kw) -> Detector:
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}', use one of {sorted(METHODS)}")
    return METHODS[method](**kw)
