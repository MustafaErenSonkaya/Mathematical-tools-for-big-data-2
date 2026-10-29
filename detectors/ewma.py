"""EWMA control chart (Roberts, 1959).

In sigma0 units:
    Z_t = lambda * z_t + (1 - lambda) * Z_{t-1},  Z_0 = 0     (i.e. Z_0 = mu0)
    limits  +- L * sqrt(lambda/(2-lambda) * [1 - (1-lambda)^(2t)])
which is the slide formula mu0 +- L sigma0 sqrt(...) divided by sigma0.

The variance of Z_t is computed with the recursion
    v_t = (1 - lambda_t)^2 v_{t-1} + lambda_t^2,   v_0 = 0
which gives exactly the closed form above for a constant lambda and also
works for the irregular time variant lambda_t = 1 - exp(-rho * dt), used when
rho > 0 and update() gets a time stamp t.

After an alarm Z and t restart (the limits start narrow again).
"""
from __future__ import annotations

import math

import numpy as np

from .base import Detector, in_half_open01, positive, side_of


def _rho(v):
    return None if v >= 0 else "must be >= 0 (0 = regular time, use lambda)"


class EWMA(Detector):
    name = "ewma"
    PARAMS = {"lam": (0.1, in_half_open01),  # smoothing weight lambda
              "L": (3.0, positive),          # limit width, in std of Z
              "rho": (0.0, _rho)}            # irregular time decay rate (0 = off)

    def _init_state(self, shape):
        self.z = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t_last = None

    def _reset_state(self, mask):
        self.z = np.where(mask, 0.0, self.z)
        self.v = np.where(mask, 0.0, self.v)

    def _weight(self, t):
        rho = self._params["rho"]
        if rho > 0 and t is not None:
            dt = 1.0 if self.t_last is None else float(t) - self.t_last
            self.t_last = float(t)
            return 1.0 - math.exp(-rho * max(dt, 0.0))
        return self._params["lam"]

    def _step(self, z, t):
        lam = self._weight(t)
        self.z = lam * z + (1 - lam) * self.z
        self.v = (1 - lam) ** 2 * self.v + lam ** 2
        lim = self._params["L"] * np.sqrt(self.v)
        up, down = self.z > lim, self.z < -lim
        return dict(statistic=self.z.copy(), threshold_up=lim, threshold_down=-lim,
                    alarm=up | down, side=side_of(up, down))


def ewma_limit(lam: float, L: float, t: int) -> float:
    """Closed-form finite-time limit (sigma0 units), as on the slide."""
    return L * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
