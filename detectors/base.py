"""Shared machinery for the sequential detectors.

Every detector has the same interface, so the live app, the calibration and
the evaluation can treat them all the same way:

    det.reset()                       forget everything, start a new run
    out = det.update(x, t=None)       feed one new observation
    det.params                        current hyperparameters (dict)
    det.set_params(**kw)              change hyperparameters (validated)

`update` returns a dict with at least
    statistic   the monitored quantity (or "statistic_up"/"statistic_down")
    threshold   the alarm limit(s)
    alarm       True when the detector raises an alarm at this step
    side        "up", "down" or None (which way the change went)

Units: the univariate detectors work on the standardised value
z_t = (X_t - mu0) / sigma0, so every statistic and threshold is in sigma0
units. This makes one setting mean the same thing for a temperature in °R and
a speed in rpm, and it is the unit the injector uses for anomaly sizes.

Vectorised: x may be a scalar (one sensor) or a numpy array of any shape
(for example engines x sensors). Each element is an independent detector
with its own state; this is how the evaluation runs 100 engines x 14 sensors
in one call. With a scalar input the outputs are plain Python scalars.
"""
from __future__ import annotations

import numpy as np

RESTART_POLICIES = ("reset", "continue")


class Detector:
    """Base class. Subclasses define PARAMS (name -> (default, check)) and
    implement _reset_state(mask) and _step(z, t)."""

    name = "base"
    PARAMS: dict = {}
    multivariate = False

    def __init__(self, mu0=0.0, sigma0=1.0, **params):
        self.mu0 = np.asarray(mu0, dtype=float)
        self.sigma0 = np.asarray(sigma0, dtype=float)
        if np.any(self.sigma0 <= 0):
            raise ValueError("sigma0 must be > 0")
        self._params = {k: v[0] for k, v in self.PARAMS.items()}
        self._params.update(restart="reset", clip=None)
        self.set_params(**params)
        self.reset()

    # ---- parameters -------------------------------------------------------
    @property
    def params(self) -> dict:
        return dict(self._params)

    def set_params(self, **kw):
        """Change hyperparameters. Raises ValueError with a readable message."""
        new = dict(self._params)
        for k, v in kw.items():
            if k not in new:
                raise ValueError(f"{self.name}: unknown parameter '{k}'; allowed {sorted(new)}")
            new[k] = v
        for k, (_, check) in self.PARAMS.items():
            new[k] = float(new[k])
            msg = check(new[k])
            if msg:
                raise ValueError(f"{self.name}.{k} = {new[k]}: {msg}")
        if new["restart"] not in RESTART_POLICIES:
            raise ValueError(f"{self.name}.restart must be one of {RESTART_POLICIES}")
        if new["clip"] is not None:
            new["clip"] = float(new["clip"])
            if new["clip"] <= 0:
                raise ValueError(f"{self.name}.clip must be > 0 (or None for no clipping)")
        old, self._params = getattr(self, "_params", None), new
        try:
            self._validate_extra()
        except ValueError:
            self._params = old              # a refused change leaves the detector as it was
            raise

    def _validate_extra(self):
        pass

    def set_baseline(self, mu0, sigma0):
        """Set the model parameters (estimated from the healthy baseline)."""
        self.mu0 = np.asarray(mu0, dtype=float)
        self.sigma0 = np.asarray(sigma0, dtype=float)
        if np.any(self.sigma0 <= 0):
            raise ValueError("sigma0 must be > 0")

    # ---- running ------------------------------------------------------------
    def reset(self):
        self._shape = None
        self._state_ready = False

    def standardise(self, x):
        z = (np.asarray(x, dtype=float) - self.mu0) / self.sigma0
        c = self._params["clip"]
        if c is not None:
            z = np.clip(z, -c, c)
        return z

    def update(self, x, t=None) -> dict:
        scalar = np.ndim(x) == 0
        z = self.standardise(x)
        if not self._state_ready or self._shape != z.shape:
            self._shape = z.shape
            self._init_state(z.shape)
            self._state_ready = True
        out = self._step(z, t)
        alarm = out["alarm"]
        if self._params["restart"] == "reset" and np.any(alarm):
            self._reset_state(alarm)
        if scalar:
            out = {k: _scalar(v) for k, v in out.items()}
            if out.get("side") == "":
                out["side"] = None
        return out

    # subclasses
    def _init_state(self, shape):
        raise NotImplementedError

    def _reset_state(self, mask):
        raise NotImplementedError

    def _step(self, z, t):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self._params})"


def side_of(up, down):
    """String array 'up' / 'down' / '' from two boolean arrays."""
    return np.where(up, "up", np.where(down, "down", ""))


def _scalar(v):
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    elif isinstance(v, np.generic):
        v = v.item()
    return v


def positive(v):
    return None if v > 0 else "must be > 0"


def in_open01(v):
    return None if 0 < v < 1 else "must be between 0 and 1 (exclusive)"


def in_half_open01(v):
    return None if 0 < v <= 1 else "must satisfy 0 < value <= 1"
