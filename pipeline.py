"""Glue shared by calibrate.py, evaluate.py and the live app.

- load_params():   params.yaml with "auto" thresholds filled in from calibration.json
- Setup:           the data, the sensors, and the model parameters estimated from it
                   (fleet sigma0, MEWMA covariance, wear directions)
- to_panel():      engines x cycles x sensors array (NaN after an engine's last cycle)
- run_method():    run one detector over all engines at once, return the alarms
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cmapss import fleet_sigma0, informative_sensors, load_cmapss, wear_directions
from detectors import SHAPE_PARAM, THRESHOLD_PARAM, make

ROOT = Path(__file__).resolve().parent
METHOD_ORDER = ["shewhart", "sprt", "cusum", "ewma", "mewma"]
LABELS = {"shewhart": "Shewhart", "sprt": "SPRT", "cusum": "CUSUM", "ewma": "EWMA", "mewma": "MEWMA"}
COLORS = {"shewhart": "#7f7f7f", "sprt": "#9467bd", "cusum": "#1f77b4", "ewma": "#ff7f0e", "mewma": "#2ca02c"}


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_calibration(path=ROOT / "calibration.json"):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def calibrated_threshold(cal, method, shape_value=None, target=None, source=None, extra=None):
    """Threshold for `method` from calibration.json, interpolated linearly
    between the calibrated shape values (kappa, lambda or delta).
    extra: for SPRT, {"beta": b} (nearest calibrated beta is used)."""
    if cal is None:
        raise ValueError("calibration.json not found: run  python calibrate.py  first")
    target = int(target or cal["target_arl0"])
    source = source or cal["default_source"]
    rows = [r for r in cal["results"][method] if r["source"] == source]
    if not rows:
        raise ValueError(f"no calibration for {method} / {source}")
    if method == "sprt":
        betas = sorted({r["beta"] for r in rows})
        b = min(betas, key=lambda v: abs(v - float((extra or {}).get("beta", 0.1))))
        rows = [r for r in rows if r["beta"] == b]
    key = str(target)
    if key not in rows[0]["thresholds"]:
        raise ValueError(f"target ARL0 {target} not calibrated; available {sorted(rows[0]['thresholds'])}")
    sp = SHAPE_PARAM[method]
    if sp is None:
        return rows[0]["thresholds"][key]
    rows = sorted(rows, key=lambda r: r[sp])
    xs = np.array([r[sp] for r in rows])
    ys = np.array([r["thresholds"][key] for r in rows])
    v = float(shape_value)
    if method == "sprt":     # alpha spans decades: interpolate log(alpha)
        return float(np.exp(np.interp(v, xs, np.log(ys))))
    return float(np.interp(v, xs, ys))


def calibrated_shape_range(cal, method, source=None):
    """(lowest, highest) calibrated kappa / lambda / delta, or None for Shewhart.
    Outside this range calibrated_threshold() can only reuse the nearest edge."""
    sp = SHAPE_PARAM[method]
    if sp is None or cal is None:
        return None
    vals = [r[sp] for r in cal["results"][method] if r["source"] == (source or cal["default_source"])]
    return (min(vals), max(vals)) if vals else None


def resolve_detector_params(det_cfg: dict, cal, target=None, source=None) -> dict:
    """Replace every "auto" threshold by its calibrated value."""
    out = copy.deepcopy(det_cfg)
    for m, p in out.items():
        tp = THRESHOLD_PARAM[m]
        if p.get(tp) == "auto":
            sp = SHAPE_PARAM[m]
            v = calibrated_threshold(cal, m, p.get(sp) if sp else None, target, source,
                                     extra={"beta": p.get("beta", 0.1)})
            p[tp] = float(f"{v:.6g}")
    return out


def load_params(path=ROOT / "params.yaml", calibration=ROOT / "calibration.json", resolve=True) -> dict:
    p = load_yaml(path)
    if resolve:
        cal = load_calibration(calibration)
        c = p["calibration"]
        p["detectors"] = resolve_detector_params(p["detectors"], cal, c["target_arl0"], c["source"])
    return p


def detector_kwargs(params: dict, method: str) -> dict:
    kw = dict(params["detectors"][method])
    kw.update(restart=params.get("restart", "reset"), clip=params.get("clip"))
    return kw


# --------------------------------------------------------------------------
# data and model parameters
# --------------------------------------------------------------------------
@dataclass
class Setup:
    df: pd.DataFrame
    sensors: list
    baseline_cycles: int
    sigma0_fleet: dict
    wear: dict
    cov: np.ndarray            # p x p correlation of standardised baseline noise

    @property
    def sigma0_vec(self):
        return np.array([self.sigma0_fleet[s] for s in self.sensors])


def fleet_cov(df, sensors, baseline_cycles, sigma0) -> np.ndarray:
    """Pooled within-engine covariance of z = (x - engine mean) / sigma0 over
    the baseline windows of all engines. Diagonal is ~1 by construction."""
    head = df[df["cycle"] <= baseline_cycles]
    s0 = np.array([sigma0[s] for s in sensors])
    acc, dof = np.zeros((len(sensors), len(sensors))), 0
    for _, g in head.groupby("unit"):
        z = (g[sensors].to_numpy(float) - g[sensors].to_numpy(float).mean(0)) / s0
        acc += z.T @ z
        dof += len(g) - 1
    return acc / dof


def load_setup(params=None, data=None) -> Setup:
    params = params or load_params(resolve=False)
    df = load_cmapss(str(ROOT / (data or params["scenario"]["data"])))
    sens = params["scenario"].get("sensors", "all")
    sensors = informative_sensors(df) if sens in (None, "all") else list(sens)
    B = int(params["baseline_cycles"])
    s0 = fleet_sigma0(df, sensors, B)
    return Setup(df=df, sensors=sensors, baseline_cycles=B, sigma0_fleet=s0,
                 wear=wear_directions(df, sensors), cov=fleet_cov(df, sensors, B, s0))


# --------------------------------------------------------------------------
# panels and running
# --------------------------------------------------------------------------
def to_panel(df, sensors, value_cols=None):
    """Stack engines into an array X[engine, step, sensor] (NaN padded).
    value_cols: columns to read instead of the sensor names (e.g. clean copies)."""
    cols = value_cols or sensors
    units = sorted(df["unit"].unique())
    groups = {u: g.sort_values("cycle") for u, g in df.groupby("unit")}
    T = max(len(g) for g in groups.values())
    X = np.full((len(units), T, len(sensors)), np.nan)
    cyc = np.full((len(units), T), -1, dtype=int)
    lengths = np.zeros(len(units), dtype=int)
    for i, u in enumerate(units):
        g = groups[u]
        X[i, :len(g)] = g[cols].to_numpy(float)
        cyc[i, :len(g)] = g["cycle"].to_numpy()
        lengths[i] = len(g)
    return dict(X=X, cycles=cyc, lengths=lengths, units=np.array(units))


def baseline_stats(X, B, sigma0_fleet_vec=None, sigma_source="fleet"):
    """mu0 per engine and sensor from the first B steps; sigma0 fleet or per engine."""
    head = X[:, :B, :]
    mu0 = head.mean(axis=1)
    if sigma_source == "engine":
        sig = head.std(axis=1, ddof=1)
        sig = np.where(sig > 0, sig, 1e-9)
    else:
        sig = np.broadcast_to(sigma0_fleet_vec, mu0.shape)
    return mu0, sig


def run_method(method, kw, X, mu0, sigma0, B, cov=None, record=False):
    """Run one method over all engines in parallel, monitoring steps B..T-1.

    Returns alarms: bool [E, T, p], one per sensor. MEWMA has one alarm per
    step; it is put on its top_sensor (largest smoothed deviation), so the
    "any sensor" view is exactly the MEWMA alarm.
    With restart="continue" only the first step of each alarm period counts.
    record=True also returns the main statistic per step."""
    E, T, p = X.shape
    det = make(method, mu0=mu0, sigma0=sigma0, **kw)
    mv = method == "mewma"
    if mv:
        det.set_cov(cov if cov is not None else np.eye(p))
    alarms = np.zeros((E, T, p), dtype=bool)
    stats = np.full((E, T) if mv else (E, T, p), np.nan) if record else None
    rows = np.arange(E)
    with np.errstate(invalid="ignore"):
        for t in range(B, T):
            out = det.update(X[:, t, :], t=t)
            if mv:
                alarms[rows, t, out["top_sensor"]] = out["alarm"]
            else:
                alarms[:, t] = out["alarm"]
            if record:
                stats[:, t] = _main_stat(method, out)
    if kw.get("restart") == "continue":
        prev = np.zeros_like(alarms)
        prev[:, 1:] = alarms[:, :-1]
        alarms = alarms & ~prev
    return (alarms, stats) if record else alarms


def _main_stat(method, out):
    if "statistic" in out:
        return out["statistic"]
    return np.maximum(out["statistic_up"], out["statistic_down"])


def system_alarms(alarms):
    """'any sensor alarms' rule: [E, T, p] -> [E, T]."""
    return alarms.any(axis=2)


def remove_wear(df, sensors, baseline_cycles, degree=3):
    """Wear-free copy of the data (for a robustness check only): for each engine
    and sensor, fit a cubic trend in cycle over the whole life and subtract it,
    keeping the baseline mean. The real noise (rounding, tails, dependence)
    stays; the slow degradation goes."""
    out = df.copy()
    out[sensors] = out[sensors].astype(float)
    for _, g in df.groupby("unit"):
        t = g["cycle"].to_numpy(float)
        tn = (t - t.mean()) / t.std()
        V = np.vander(tn, degree + 1)
        for s in sensors:
            y = g[s].to_numpy(float)
            trend = V @ np.linalg.lstsq(V, y, rcond=None)[0]
            out.loc[g.index, s] = y - trend + trend[:baseline_cycles].mean()
    return out


def clean_mask(panel, B, life_fraction):
    """Monitored steps that are in the healthy part of life: B <= step < round(f*length)."""
    E, T = panel["cycles"].shape
    idx = np.arange(T)[None, :]
    last = np.round(panel["lengths"] * life_fraction).astype(int)[:, None]
    return (idx >= B) & (idx < last)
