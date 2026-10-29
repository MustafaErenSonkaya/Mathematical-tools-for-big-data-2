"""Helpers for loading the NASA C-MAPSS turbofan engine data (FD001..FD004).

Each row is one operating cycle (one flight) of one engine ("unit").
Columns: unit, cycle, 3 operational settings, 21 sensors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS

# Symbol, description, unit (Saxena et al. 2008, Table 2)
SENSOR_INFO = {
    "s1": ("T2", "Total temperature at fan inlet", "°R"),
    "s2": ("T24", "Total temperature at LPC outlet", "°R"),
    "s3": ("T30", "Total temperature at HPC outlet", "°R"),
    "s4": ("T50", "Total temperature at LPT outlet", "°R"),
    "s5": ("P2", "Pressure at fan inlet", "psia"),
    "s6": ("P15", "Total pressure in bypass-duct", "psia"),
    "s7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s8": ("Nf", "Physical fan speed", "rpm"),
    "s9": ("Nc", "Physical core speed", "rpm"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)", "-"),
    "s11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", "-"),
    "s16": ("farB", "Burner fuel-air ratio", "-"),
    "s17": ("htBleed", "Bleed enthalpy", "-"),
    "s18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}


def sensor_label(s: str) -> str:
    sym, desc, unit = SENSOR_INFO.get(s, (s, s, ""))
    return f"{s} ({sym}, {unit})"


def load_cmapss(path: str) -> pd.DataFrame:
    """Read a train_FDxxx.txt / test_FDxxx.txt file into a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def informative_sensors(df: pd.DataFrame) -> list[str]:
    """Sensors that actually move. Constant or two-level sensors are dropped
    (in FD001 that removes s1, s5, s6, s10, s16, s18, s19)."""
    return [s for s in SENSOR_COLS if df[s].nunique() > 2 and df[s].std() > 0]


def wear_directions(df: pd.DataFrame, sensors: list[str]) -> dict[str, int]:
    """+1 if the sensor rises as the engine wears out, -1 if it falls.

    Measured as the sign of the correlation between the sensor and the
    fraction of life used (cycle / last cycle of that unit). Only meaningful
    on a *train* file, where every engine runs to failure."""
    life = df["cycle"] / df.groupby("unit")["cycle"].transform("max")
    out = {}
    for s in sensors:
        r = np.corrcoef(df[s], life)[0, 1]
        out[s] = 1 if (np.isnan(r) or r >= 0) else -1
    return out


def fleet_sigma0(df: pd.DataFrame, sensors: list[str], baseline_cycles: int) -> dict[str, float]:
    """Baseline noise level sigma0 per sensor.

    For each engine we take its first `baseline_cycles` cycles (the healthy
    part), compute the variance inside that window, and average the variances
    over all engines. Averaging *within-engine* variances avoids counting the
    level differences between engines (initial wear) as noise."""
    head = df[df["cycle"] <= baseline_cycles]
    var = head.groupby("unit")[sensors].var(ddof=1).mean()
    return {s: float(np.sqrt(var[s])) for s in sensors}
