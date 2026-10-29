"""Wald's Sequential Probability Ratio Test, used as a streaming detector.

Gaussian model in sigma0 units: f0 = N(0, 1), f1 = N(+-Delta, 1).
    l_t = log f1(z_t)/f0(z_t) = Delta * (z_t - Delta/2)        (upward test)
    S_t = S_{t-1} + l_t
    A = (1 - beta)/alpha,  B = beta/(1 - alpha)                (Wald limits)
    S_t >= log A  -> decide H1: raise an alarm, restart
    S_t <= log B  -> decide H0: restart quietly
    otherwise     -> keep sampling

Two one-sided tests run side by side (shift up and shift down), each
restarting on its own decisions.

Wald's limits are approximations (they ignore the overshoot over the
boundary), and alpha is the error of ONE test, not the false alarm rate of
the repeated test. That is why alpha is calibrated to a target ARL0 instead
of being chosen directly.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Detector, in_open01, positive, side_of


class SPRT(Detector):
    name = "sprt"
    PARAMS = {"delta": (1.0, positive),   # design shift Delta, sigma0 units
              "alpha": (0.01, in_open01), # type I error of one test
              "beta": (0.1, in_open01)}   # type II error of one test

    def _validate_extra(self):
        if self._params["alpha"] + self._params["beta"] >= 1:
            raise ValueError("sprt: alpha + beta must be < 1 (otherwise log A <= log B)")

    @property
    def log_a(self):
        p = self._params
        return math.log((1 - p["beta"]) / p["alpha"])

    @property
    def log_b(self):
        p = self._params
        return math.log(p["beta"] / (1 - p["alpha"]))

    def _init_state(self, shape):
        self.s_up = np.zeros(shape)
        self.s_down = np.zeros(shape)

    def _reset_state(self, mask):
        self.s_up = np.where(mask, 0.0, self.s_up)
        self.s_down = np.where(mask, 0.0, self.s_down)

    def _step(self, z, t):
        d = self._params["delta"]
        la, lb = self.log_a, self.log_b
        self.s_up = self.s_up + d * (z - d / 2)
        self.s_down = self.s_down + d * (-z - d / 2)
        up, down = self.s_up >= la, self.s_down >= la
        acc_up, acc_down = self.s_up <= lb, self.s_down <= lb
        out = dict(statistic_up=self.s_up.copy(), statistic_down=self.s_down.copy(),
                   threshold_upper=np.full(z.shape, la), threshold_lower=np.full(z.shape, lb),
                   alarm=up | down, side=side_of(up, down), accept_h0=acc_up | acc_down)
        # quiet restart of a test that accepted H0 (the alarm restart is done by the base class)
        self.s_up = np.where(acc_up, 0.0, self.s_up)
        self.s_down = np.where(acc_down, 0.0, self.s_down)
        return out
