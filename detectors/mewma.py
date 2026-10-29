"""Multivariate EWMA (Lowry et al., 1992).

On the p-vector of standardised sensors z_t:
    Z_t   = lambda * z_t + (1 - lambda) * Z_{t-1},  Z_0 = 0
    S_Z,t = lambda/(2-lambda) * [1 - (1-lambda)^(2t)] * Sigma
    T2_t  = Z_t' S_Z,t^{-1} Z_t
    alarm when T2_t > h, then Z and t restart.

Sigma is the covariance of the standardised baseline noise (a correlation
matrix when sigma0 is the per-sensor std). There is one statistic for all
sensors: the alarm says "something changed". As a simple diagnosis the output
also names top_sensor, the sensor with the largest |Z_j| / sqrt(Sigma_jj).

The input of update() has the sensors on the LAST axis: shape (p,) for one
engine, (n_engines, p) for many engines in parallel.
"""
from __future__ import annotations

import numpy as np

from .base import Detector, _scalar, in_half_open01, positive


class MEWMA(Detector):
    name = "mewma"
    multivariate = True
    PARAMS = {"lam": (0.1, in_half_open01),
              "h": (30.0, positive)}

    def __init__(self, mu0=0.0, sigma0=1.0, cov=None, **params):
        self.cov = None
        super().__init__(mu0=mu0, sigma0=sigma0, **params)
        if cov is not None:
            self.set_cov(cov)

    def set_cov(self, cov):
        cov = np.asarray(cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("mewma: cov must be a square p x p matrix")
        self.cov = cov
        self.cov_inv = np.linalg.inv(cov)
        self.reset()

    def _init_state(self, shape):
        if self.cov is None:
            self.set_cov(np.eye(shape[-1]))
        if self.cov.shape[0] != shape[-1]:
            raise ValueError(f"mewma: cov is {self.cov.shape[0]}x{self.cov.shape[0]} "
                             f"but the input has {shape[-1]} sensors")
        self.z = np.zeros(shape)
        self.n = np.zeros(shape[:-1])

    def _reset_state(self, mask):
        mask = np.asarray(mask)
        self.z = np.where(mask[..., None], 0.0, self.z)
        self.n = np.where(mask, 0, self.n)

    def _step(self, z, t):
        lam, h = self._params["lam"], self._params["h"]
        self.z = lam * z + (1 - lam) * self.z
        self.n = self.n + 1
        c = lam / (2 - lam) * (1 - (1 - lam) ** (2 * self.n))
        t2 = np.einsum("...i,ij,...j->...", self.z, self.cov_inv, self.z) / c
        alarm = t2 > h
        # simple diagnosis: the sensor whose smoothed value is furthest from 0
        contrib = np.abs(self.z) / np.sqrt(np.diag(self.cov))
        return dict(statistic=t2, threshold=np.full(t2.shape, h), alarm=alarm,
                    side=np.full(t2.shape, ""), top_sensor=np.argmax(contrib, axis=-1),
                    z_smoothed=self.z.copy())

    def update(self, x, t=None) -> dict:
        """x has the p sensors on its last axis. One engine: shape (p,), and
        the outputs are plain scalars."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            raise ValueError("mewma needs a vector of sensors")
        out = super().update(x, t)
        if x.ndim == 1:
            out = {k: (v if k == "z_smoothed" else _scalar(v)) for k, v in out.items()}
            out["side"] = None
        return out
