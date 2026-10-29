"""Shewhart chart: the simplest reference detector.

Alarm when one observation is far from the baseline mean:
    |Y_t - mu0| > k * sigma0
It has no memory, so it reacts to a single extreme sample (a spike) and is
slow for small persistent shifts. It is here as a baseline: it measures
"anomaly" (outlier), while CUSUM / EWMA / SPRT measure "change".
"""
from __future__ import annotations

import numpy as np

from .base import Detector, positive, side_of


class Shewhart(Detector):
    name = "shewhart"
    PARAMS = {"k": (4.0, positive)}      # limit in sigma0 units

    def _init_state(self, shape):
        pass

    def _reset_state(self, mask):
        pass                              # no memory, nothing to reset

    def _step(self, z, t):
        k = self._params["k"]
        up, down = z > k, z < -k
        return dict(statistic=z, threshold_up=np.full(z.shape, k), threshold_down=np.full(z.shape, -k),
                    alarm=up | down, side=side_of(up, down))
