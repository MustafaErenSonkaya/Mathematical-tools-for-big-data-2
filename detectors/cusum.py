"""Two-sided CUSUM (Page, 1954).

In sigma0 units (z_t = (Y_t - mu0) / sigma0):
    C+_t = max(0, C+_{t-1} + z_t - kappa)
    C-_t = max(0, C-_{t-1} - z_t - kappa)
    C+_0 = C-_0 = 0
    alarm when C+_t > h or C-_t > h, then both sums restart at 0.

This is the slide formula with Y - mu0 - kappa*sigma0 divided by sigma0, so
kappa and h are in sigma0 units. kappa ~ Delta/2 (Delta = the shift we want
to catch fastest) is a heuristic that comes from the SPRT: with
f0 = N(0,1), f1 = N(Delta,1) the log likelihood ratio is
Delta * (z - Delta/2), so C+ is (1/Delta) times a repeated SPRT whose lower
boundary is reset at 0 instead of stopping at log B.
"""
from __future__ import annotations

import numpy as np

from .base import Detector, positive, side_of


def _nonneg(v):
    return None if v >= 0 else "must be >= 0"


class CUSUM(Detector):
    name = "cusum"
    PARAMS = {"kappa": (0.5, _nonneg),   # reference value (allowance), sigma0 units
              "h": (5.0, positive)}      # decision interval, sigma0 units

    def _init_state(self, shape):
        self.c_up = np.zeros(shape)
        self.c_down = np.zeros(shape)

    def _reset_state(self, mask):
        self.c_up = np.where(mask, 0.0, self.c_up)
        self.c_down = np.where(mask, 0.0, self.c_down)

    def _step(self, z, t):
        k, h = self._params["kappa"], self._params["h"]
        self.c_up = np.maximum(0.0, self.c_up + z - k)
        self.c_down = np.maximum(0.0, self.c_down - z - k)
        up, down = self.c_up > h, self.c_down > h
        return dict(statistic_up=self.c_up.copy(), statistic_down=self.c_down.copy(),
                    threshold=np.full(z.shape, h), alarm=up | down, side=side_of(up, down))
