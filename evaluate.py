"""Usefulness evaluation: experiments E1..E7 on C-MAPSS FD001 with injected anomalies.

Every experiment uses the injector in batch mode with a fixed seed, runs all
detectors with the calibrated defaults from params.yaml, and scores per
event and per sensor:

  detected      an alarm on that sensor in [start, end + grace]
                (MEWMA has one statistic; its alarm is put on the sensor with
                the largest smoothed deviation and scored like the others;
                E5 scores every method at system level instead)
  delay         first alarm cycle - start (detected events only)
  false alarm   an alarm outside every event window (+ grace) of that sensor,
                counted only in the healthy part of life (<= clean_life_fraction)
  placebo rate  the same detection rule on the SAME windows but on the clean
                data (no injection): how often an alarm falls in the window by
                chance. A detection rate close to the placebo rate means the
                method does not see the event.

Outputs: reports/results.md, reports/tables/*.csv, reports/figures/*.png,
reports/results_settings.yaml (everything needed to reproduce).

    python evaluate.py               # all experiments
    python evaluate.py E1 E3         # only some
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from injector import AnomalyInjector, InjectorConfig, inject_frame  # noqa: E402
from pipeline import (COLORS, LABELS, METHOD_ORDER, ROOT, baseline_stats, calibrated_threshold,  # noqa: E402
                      detector_kwargs, load_calibration, load_params, load_setup, load_yaml,
                      remove_wear, resolve_detector_params, run_method, to_panel)

REP = ROOT / "reports"
FIG = REP / "figures"
TAB = REP / "tables"
OFF = {k: {"enabled": False} for k in ("spike", "level_shift", "drift", "noise_burst", "stuck")}
PLOT_STYLE = {"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
              "axes.spines.right": False, "font.size": 10}
plt.rcParams.update(PLOT_STYLE)


# --------------------------------------------------------------------------
# context shared by all experiments
# --------------------------------------------------------------------------
class Ctx:
    def __init__(self):
        self.params = load_params()
        self.cal = load_calibration()
        self.setup = load_setup(self.params)
        self.B = self.setup.baseline_cycles
        self.grace = int(self.params["scoring"]["grace"])
        self.clean_f = float(self.params["scoring"]["clean_life_fraction"])
        self.early_f = float(self.params["calibration"].get("early_life_fraction", 0.3))
        last = self.setup.df.groupby("unit")["cycle"].transform("max")
        self.df_clean = self.setup.df[self.setup.df["cycle"] <= (last * self.clean_f).round()]
        nowear = remove_wear(self.setup.df, self.setup.sensors, self.B)
        self.df_nowear = nowear[nowear["cycle"] <= (last * self.clean_f).round()]
        # the clean panel keeps every engine's full length for the false-alarm zone
        self.full_len = self.setup.df.groupby("unit")["cycle"].max().sort_index().to_numpy()
        self.log = {}

    def kw(self, method, params=None, **over):
        kw = detector_kwargs(params or self.params, method)
        kw.update(over)
        return kw

    def real_params(self):
        """Same shapes, thresholds calibrated on the real data up to 60% of life."""
        p = copy.deepcopy(self.params)
        raw = load_params(resolve=False)["detectors"]
        p["detectors"] = resolve_detector_params(raw, self.cal, source="real")
        return p


def injector_cfg(seed, mode="independent", B=30, **kinds):
    k = copy.deepcopy(OFF)
    for name, spec in kinds.items():
        k[name] = dict(spec, enabled=True)
    return {"seed": seed, "baseline_cycles": B, "min_gap": 10, "mode": mode, "kinds": k}


def make_scenario(ctx, cfg, inj_sensors=None, df=None, B=None):
    """Inject into the healthy part of each engine. Returns a dict with the
    injected panel, the clean panel, the events and masks."""
    sensors = ctx.setup.sensors
    inj_sensors = inj_sensors or sensors
    df = ctx.df_clean if df is None else df
    inj = AnomalyInjector(inj_sensors, InjectorConfig.from_dict(cfg),
                          sigma0={s: ctx.setup.sigma0_fleet[s] for s in inj_sensors},
                          wear_dir=ctx.setup.wear)
    data, events = inject_frame(df, inj_sensors, inj)
    pan = to_panel(data, sensors)
    clean = to_panel(data, sensors, [f"{s}_clean" if s in inj_sensors else s for s in sensors])
    fa = fa_mask(ctx, pan, ctx.clean_f, B)
    early = fa_mask(ctx, pan, ctx.early_f, B)
    return dict(panel=pan, clean=clean, events=events, fa=fa, early=early, cfg=cfg,
                inj_sensors=inj_sensors)


SIGNALS = ("wear removed", "raw")


def signal_df(ctx, signal):
    return ctx.df_nowear if signal == "wear removed" else ctx.df_clean


def fa_mask(ctx, pan, f, B=None):
    """Monitored steps (from the end of the baseline B) before fraction f of
    the engine's FULL life."""
    B = ctx.B if B is None else B
    E, T = pan["cycles"].shape
    idx = np.arange(T)[None, :]
    full = ctx.full_len[np.searchsorted(np.sort(ctx.setup.df["unit"].unique()), pan["units"])]
    return (idx >= B) & (idx < np.round(full * f)[:, None]) & (idx < pan["lengths"][:, None])


def run_all(ctx, panel, methods, params=None, overrides=None, B=None):
    """alarms per method on this panel (mu0 from the panel's own baseline window)."""
    B = B or ctx.B
    mu0, sig = baseline_stats(panel["X"], B, ctx.setup.sigma0_vec, "fleet")
    out = {}
    for m in methods:
        kw = ctx.kw(m, params, **((overrides or {}).get(m, {})))
        out[m] = run_method(m, kw, panel["X"], mu0, sig, B, cov=ctx.setup.cov)
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def windows(sc, grace, sensors):
    """Per-sensor window mask [E, T, p] and any-sensor mask [E, T] of all events (+grace)."""
    pan = sc["panel"]
    E, T = pan["cycles"].shape
    row = {u: i for i, u in enumerate(pan["units"])}
    col = {s: j for j, s in enumerate(sensors)}
    W = np.zeros((E, T, len(sensors)), dtype=bool)
    for ev in sc["events"].itertuples():
        W[row[ev.unit], ev.start - 1: ev.end + grace, col[ev.sensor]] = True
    return W, W.any(axis=2)


def score(sc, alarms, method, grace, sensors, system=False, kinds=None):
    """Score one method's alarms on one scenario.
    system=True: events are judged per (unit, event_id) with the any-sensor alarm
    (used for joint shifts); otherwise per (event, sensor)."""
    pan = sc["panel"]
    row = {u: i for i, u in enumerate(pan["units"])}
    col = {s: j for j, s in enumerate(sensors)}
    ev = sc["events"]
    if kinds:
        ev = ev[ev["kind"].isin(kinds)]
    sysal = alarms.any(axis=2)
    recs = []
    groups = ev.groupby(["unit", "event_id"]) if system else ev.groupby(["unit", "event_id", "sensor"])
    for key, g in groups:
        e = row[key[0]]
        s0, s1 = int(g["start"].min()) - 1, int(g["end"].max()) + grace
        a = sysal[e, s0:s1] if system else alarms[e, s0:s1, col[key[2]]]
        hit = np.flatnonzero(a)
        recs.append(dict(kind=g["kind"].iloc[0], size=float(g["magnitude_sigma"].abs().iloc[0]),
                         detected=bool(len(hit)), delay=float(hit[0]) if len(hit) else np.nan))
    r = pd.DataFrame(recs, columns=["kind", "size", "detected", "delay"])
    W, Wany = windows(sc, grace, sensors)
    fa_t = (sysal & ~Wany) if system else (alarms & ~W).any(axis=2)
    n_fa = int((fa_t & sc["fa"]).sum())
    n_cyc = int(sc["fa"].sum())
    n_fa_e = int((fa_t & sc["early"]).sum())
    n_cyc_e = int(sc["early"].sum())
    det = r["delay"].dropna()
    return dict(method=method, events=len(r), detection_rate=r["detected"].mean() if len(r) else np.nan,
                median_delay=det.median() if len(det) else np.nan,
                mean_delay=det.mean() if len(det) else np.nan,
                fa_per_1000=1000 * n_fa / n_cyc, fa_per_1000_early=1000 * n_fa_e / max(n_cyc_e, 1),
                arl0_emp=(n_cyc / n_fa) if n_fa else np.inf, _events=r)


def placebo_rate(sc, alarms_clean, grace, sensors, system=False, kinds=None):
    s = score(sc, alarms_clean, "placebo", grace, sensors, system, kinds)
    return s["detection_rate"]


def evaluate(ctx, sc, methods, params=None, overrides=None, system=False, kinds=None, placebo=True,
             label=None):
    al = run_all(ctx, sc["panel"], methods, params, overrides)
    alc = run_all(ctx, sc["clean"], methods, params, overrides) if placebo else {}
    rows = []
    for m in methods:
        s = score(sc, al[m], m, ctx.grace, ctx.setup.sensors, system, kinds)
        if placebo:
            s["placebo_rate"] = placebo_rate(sc, alc[m], ctx.grace, ctx.setup.sensors, system, kinds)
        s.pop("_events")
        if label:
            s["variant"] = label
        rows.append(s)
    return rows


def fmt_table(df, cols, floatfmt=None):
    floatfmt = floatfmt or {}
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("-")
                elif np.isinf(v):
                    cells.append("inf")
                else:
                    cells.append(floatfmt.get(c, "{:.2f}").format(v))
            else:
                cells.append(str(LABELS.get(v, v)) if c == "method" else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


METRIC_COLS = ["detection_rate", "placebo_rate", "median_delay", "mean_delay", "fa_per_1000",
               "fa_per_1000_early", "arl0_emp"]
FMT = {"detection_rate": "{:.0%}", "placebo_rate": "{:.0%}", "median_delay": "{:.1f}",
       "mean_delay": "{:.1f}", "fa_per_1000": "{:.1f}", "fa_per_1000_early": "{:.1f}", "arl0_emp": "{:.0f}"}


def save(df, name):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def e1(ctx):
    sizes = [0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    real = ctx.real_params()
    for d in sizes:
        sc = make_scenario(ctx, injector_cfg(101, level_shift=dict(per_100=1.0, magnitude=[d, d],
                                                                    duration=[30, 60], direction="both")))
        for r in evaluate(ctx, sc, METHOD_ORDER, label="iid_est"):
            rows.append(dict(r, shift=d))
        for r in evaluate(ctx, sc, METHOD_ORDER, params=real, label="real", placebo=False):
            rows.append(dict(r, shift=d))
        scw = make_scenario(ctx, injector_cfg(101, level_shift=dict(per_100=1.0, magnitude=[d, d],
                                                                     duration=[30, 60], direction="both")),
                            df=ctx.df_nowear)
        for r in evaluate(ctx, scw, METHOD_ORDER, label="wear removed"):
            rows.append(dict(r, shift=d))
    df = pd.DataFrame(rows)
    save(df, "E1_level_shift")
    main = df[df.variant == "iid_est"]

    nw = df[df.variant == "wear removed"]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for m in METHOD_ORDER:
        g = main[main.method == m]
        ax[0].plot(g["shift"], g["median_delay"], "o-", color=COLORS[m], label=LABELS[m])
        ax[1].plot(g["shift"], g["detection_rate"] * 100, "o-", color=COLORS[m], label=LABELS[m])
        gw = nw[nw.method == m]
        ax[0].plot(gw["shift"], gw["median_delay"], ":", color=COLORS[m])
        ax[2].plot(gw["shift"], gw["detection_rate"] * 100, "o-", color=COLORS[m], label=LABELS[m])
        gr = df[(df.variant == "real") & (df.method == m)]
        ax[3].plot(gr["shift"], gr["detection_rate"] * 100, "o--", color=COLORS[m], label=LABELS[m])
    for i, d_ in ((1, main), (2, nw)):
        pl = d_.groupby("shift")["placebo_rate"].max() * 100
        ax[i].plot(pl.index, pl.values, "k:", label="placebo (highest)")
    ax[0].set(title="Median delay (solid: raw, dotted: wear removed)", xlabel="level shift size (sigma0)",
              ylabel="delay (cycles)")
    ax[1].set(title="Detected, raw signal, default thresholds", xlabel="level shift size (sigma0)",
              ylabel="events detected (%)", ylim=(0, 105))
    ax[2].set(title="Detected, wear removed, default thresholds", xlabel="level shift size (sigma0)",
              ylabel="events detected (%)", ylim=(0, 105))
    ax[3].set(title="Detected, raw, thresholds from real data to 60%", xlabel="level shift size (sigma0)",
              ylabel="events detected (%)", ylim=(0, 105))
    ax[0].legend()
    ax[1].legend(fontsize=8)
    fig.suptitle("E1: level shifts, all methods at the same ARL0 (500 cycles, 14 sensors)")
    fig.tight_layout()
    fig.savefig(FIG / "E1_level_shift.png")
    plt.close(fig)

    txt = [fmt_table(main, ["shift", "method"] + METRIC_COLS, FMT), "",
           "Same settings on the wear-removed signal (each engine's cubic trend subtracted, real noise kept):", "",
           fmt_table(nw, ["shift", "method"] + METRIC_COLS, FMT), "",
           "Same events, thresholds calibrated on the real data up to 60% of life:", "",
           fmt_table(df[df.variant == "real"], ["shift", "method", "detection_rate", "median_delay",
                                                 "fa_per_1000", "fa_per_1000_early"], FMT)]
    s = []
    g1 = nw[nw["shift"] == 1.0].set_index("method")
    g3 = nw[nw["shift"] == 3.0].set_index("method")
    s.append(f"On the wear-removed signal, a 1 sigma0 shift is detected by CUSUM in {g1.loc['cusum','detection_rate']:.0%} "
             f"of the events, SPRT {g1.loc['sprt','detection_rate']:.0%}, EWMA {g1.loc['ewma','detection_rate']:.0%}, "
             f"MEWMA {g1.loc['mewma','detection_rate']:.0%}, but Shewhart only {g1.loc['shewhart','detection_rate']:.0%}: "
             "methods with memory win for small shifts.")
    s.append(f"For a 3 sigma0 shift every method detects about {g3.detection_rate.median():.0%} and Shewhart is as fast "
             f"as the others (median {g3.loc['shewhart','median_delay']:.0f} cycles, CUSUM "
             f"{g3.loc['cusum','median_delay']:.0f}). CUSUM and SPRT give almost the same numbers, as expected for a "
             "repeated SPRT.")
    gm = main[main["shift"] == 1.0].set_index("method")
    s.append(f"On the raw signal the same events are partly hidden by early wear: CUSUM detects "
             f"{gm.loc['cusum','detection_rate']:.0%}, but {gm.loc['cusum','placebo_rate']:.0%} of the windows "
             "would contain an alarm even without injection (placebo).")
    gw = nw[nw["shift"] == 1.0].set_index("method")
    s.append(f"On the wear-removed signal the false alarm rate is back near the target "
             f"(CUSUM {gw.loc['cusum','fa_per_1000']:.1f} per 1000 cycles, target 2) and the placebo rate falls to "
             f"{gw.loc['cusum','placebo_rate']:.0%}, so most of the false alarms on the raw signal are early wear.")
    gr = df[(df.variant == "real") & (df["shift"] == 1.0)].set_index("method")
    s.append(f"With the thresholds calibrated on the real data up to 60% of life (they treat early wear as noise), "
             f"CUSUM detects only {gr.loc['cusum','detection_rate']:.0%} of the 1 sigma0 shifts "
             f"(EWMA {gr.loc['ewma','detection_rate']:.0%}), but its false alarm rate in that zone is "
             f"{gr.loc['cusum','fa_per_1000']:.1f} per 1000 cycles instead of "
             f"{main[(main['shift']==1.0)&(main.method=='cusum')].fa_per_1000.iloc[0]:.1f}.")
    return "E1 Delay vs level shift size", txt, s


def e2(ctx):
    sizes = [1.0, 2.0, 3.0, 5.0]
    lams = [0.05, 0.1, 0.2, 0.45]
    rows = []
    for sig in SIGNALS:
        for d in sizes:
            sc = make_scenario(ctx, injector_cfg(202, drift=dict(per_100=1.0, magnitude=[d, d], duration=[40, 60],
                                                                hold=[10, 20], direction="both")),
                               df=signal_df(ctx, sig))
            for lam in lams:
                L = calibrated_threshold(ctx.cal, "ewma", lam, source=ctx.params["calibration"]["source"])
                for r in evaluate(ctx, sc, ["ewma"], overrides={"ewma": {"lam": lam, "L": L}}, label=f"lam={lam}"):
                    rows.append(dict(r, signal=sig, drift=d, lam=lam, L=L))
            for r in evaluate(ctx, sc, ["cusum", "shewhart"], label="ref"):
                rows.append(dict(r, signal=sig, drift=d, lam=np.nan, L=np.nan))
    full = pd.DataFrame(rows)
    save(full, "E2_drift")
    df = full[full.signal == "wear removed"].copy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    cmap = plt.get_cmap("Oranges")
    for i, lam in enumerate(lams):
        g = df[(df.method == "ewma") & (df.lam == lam)]
        c = cmap(0.35 + 0.6 * i / (len(lams) - 1))
        ax[0].plot(g["drift"], g["median_delay"], "o-", color=c, label=f"EWMA lambda={lam}")
        ax[1].plot(g["drift"], g["detection_rate"] * 100, "o-", color=c, label=f"EWMA lambda={lam}")
    for m in ("cusum", "shewhart"):
        g = df[df.method == m]
        ax[0].plot(g["drift"], g["median_delay"], "s--", color=COLORS[m], label=LABELS[m])
        ax[1].plot(g["drift"], g["detection_rate"] * 100, "s--", color=COLORS[m], label=LABELS[m])
    ax[0].set(title="Median detection delay", xlabel="final drift offset (sigma0), ramp 40 to 60 cycles",
              ylabel="delay (cycles)")
    ax[1].set(title="Detection rate", xlabel="final drift offset (sigma0)", ylabel="events detected (%)",
              ylim=(0, 105))
    ax[0].legend(fontsize=8)
    fig.suptitle("E2: drifts on the wear-removed signal, EWMA with different lambda (each recalibrated to ARL0 500)")
    fig.tight_layout()
    fig.savefig(FIG / "E2_drift.png")
    plt.close(fig)
    full["setting"] = np.where(full.method == "ewma", "lambda=" + full.lam.astype(str), "-")
    txt = [fmt_table(full, ["signal", "drift", "method", "setting", "detection_rate", "placebo_rate",
                            "median_delay", "fa_per_1000"], FMT)]
    s = []
    ew = df[df.method == "ewma"]
    g1 = ew[ew.drift == 1.0].set_index("lam")
    g5 = ew[ew.drift == 5.0]
    b5 = g5.loc[g5.detection_rate.idxmax()]
    sh = df[(df.method == "shewhart")].set_index("drift")
    cu = df[(df.method == "cusum")].set_index("drift")
    s.append(f"For a small drift (to 1 sigma0) a small lambda is clearly better: lambda 0.05 detects "
             f"{g1.loc[0.05,'detection_rate']:.0%}, lambda 0.45 only {g1.loc[0.45,'detection_rate']:.0%}. "
             f"For a large drift (to 5 sigma0) the choice matters little; lambda {b5.lam} is best "
             f"({b5.detection_rate:.0%}, median delay {b5.median_delay:.0f} cycles).")
    s.append(f"CUSUM behaves like EWMA with a small lambda ({cu.loc[1.0,'detection_rate']:.0%} at 1 sigma0), while Shewhart "
             f"is poor on drifts ({sh.loc[1.0,'detection_rate']:.0%} at 1 sigma0, {sh.loc[5.0,'detection_rate']:.0%} at 5): "
             "a drift is small at the start, and only methods with memory add the small steps up.")
    return "E2 Delay vs drift size and EWMA lambda", txt, s


def e3(ctx):
    rows = []
    base = evaluate(ctx, make_scenario(ctx, injector_cfg(303), df=ctx.df_nowear), METHOD_ORDER, placebo=False)
    base_rate = {r["method"]: r["fa_per_1000"] for r in base}
    variants = [("default", {}), ("clip=4", {m: {"clip": 4.0} for m in ("sprt", "cusum", "ewma", "mewma")})]
    for rate in (1.0, 5.0):
        for size in (4.0, 6.0, 8.0):
            sc = make_scenario(ctx, injector_cfg(303, spike=dict(per_100=rate, magnitude=[size, size],
                                                                 direction="both")), df=ctx.df_nowear)
            n_spk = len(sc["events"])
            n_cyc = int(sc["fa"].sum())
            for vname, ov in variants:
                al = run_all(ctx, sc["panel"], METHOD_ORDER, overrides=ov)
                for m in METHOD_ORDER:
                    a = al[m].any(axis=2)
                    n_al = int((a & sc["fa"]).sum())
                    s = score(sc, al[m], m, 0, ctx.setup.sensors)
                    rows.append(dict(rate=rate, size=size, variant=vname, method=m, spikes=n_spk,
                                     alarms_per_1000=1000 * n_al / n_cyc,
                                     extra_alarms_per_100_spikes=100 * (n_al - base_rate[m] * n_cyc / 1000) / n_spk,
                                     spikes_flagged=s["detection_rate"]))
    df = pd.DataFrame(rows)
    save(df, "E3_spikes")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for i, v in enumerate(["default", "clip=4"]):
        for m in METHOD_ORDER:
            g = df[(df.method == m) & (df.rate == 5.0) & (df.variant == v)]
            ax[i].plot(g["size"], g["spikes_flagged"] * 100, "o-", color=COLORS[m], label=LABELS[m])
        ax[i].set(title=f"Spikes flagged at the spike cycle ({v})", xlabel="spike size (sigma0)",
                  ylabel="spikes with an alarm (%)", ylim=(-3, 103))
    ax[0].legend()
    fig.suptitle("E3: spikes only (5 per 100 cycles per sensor, wear-removed signal): a change detector should stay quiet")
    fig.tight_layout()
    fig.savefig(FIG / "E3_spikes.png")
    plt.close(fig)
    txt = ["Signal: wear removed (so that the alarm counts are not mixed with wear alarms).", "",
           "Alarm rate without any injection (per 1000 cycles): "
           + ", ".join(f"{LABELS[m]} {v:.1f}" for m, v in base_rate.items()) + ".", "",
           fmt_table(df, ["rate", "size", "variant", "method", "spikes", "alarms_per_1000",
                          "extra_alarms_per_100_spikes", "spikes_flagged"],
                     {"alarms_per_1000": "{:.1f}", "extra_alarms_per_100_spikes": "{:.1f}",
                      "spikes_flagged": "{:.0%}", "rate": "{:.0f}", "size": "{:.0f}"})]
    g = df[(df.rate == 5.0) & (df["size"] == 8.0)].set_index(["variant", "method"])
    s = [f"For 8 sigma0 spikes Shewhart flags {g.loc[('default','shewhart'),'spikes_flagged']:.0%} of them: "
         f"it is an outlier detector, so it should. CUSUM flags {g.loc[('default','cusum'),'spikes_flagged']:.0%} "
         f"and EWMA {g.loc[('default','ewma'),'spikes_flagged']:.0%}, because one huge sample can push the "
         f"statistic over the limit on its own.",
         f"Clipping z at 4 sigma0 before the update (the clip hyperparameter) brings CUSUM down to "
         f"{g.loc[('clip=4','cusum'),'spikes_flagged']:.0%} and EWMA to {g.loc[('clip=4','ewma'),'spikes_flagged']:.0%}, "
         f"so the 'anomaly is not a change' point becomes measurable: memory-based detectors with a bounded "
         f"input ignore isolated outliers."]
    return "E3 Spikes only: anomaly is not change", txt, s


def e4(ctx):
    rows = []
    for kind, spec, label in [("noise_burst", dict(per_100=1.0, magnitude=[2, 2], duration=[20, 40]), "noise x2"),
                              ("noise_burst", dict(per_100=1.0, magnitude=[4, 4], duration=[20, 40]), "noise x4"),
                              ("stuck", dict(per_100=1.0, duration=[20, 40]), "stuck"),
                              ("level_shift", dict(per_100=1.0, magnitude=[2, 2], duration=[20, 40],
                                                   direction="both"), "shift 2 sigma0 (reference)")]:
        for sig in SIGNALS:
            sc = make_scenario(ctx, injector_cfg(404, **{kind: spec}), df=signal_df(ctx, sig))
            for r in evaluate(ctx, sc, METHOD_ORDER):
                rows.append(dict(r, signal=sig, scenario=label))
    full = pd.DataFrame(rows)
    save(full, "E4_noise_stuck")
    df = full[full.signal == "wear removed"]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    scen = df.scenario.unique()
    w = 0.15
    for i, m in enumerate(METHOD_ORDER):
        g = df[df.method == m].set_index("scenario").loc[scen]
        x = np.arange(len(scen)) + (i - 2) * w
        ax.bar(x, g["detection_rate"] * 100, w, color=COLORS[m], label=LABELS[m])
        ax.scatter(x, g["placebo_rate"] * 100, marker="_", s=200, color="k", zorder=3)
    ax.scatter([], [], marker="_", s=200, color="k", label="placebo (same window, no injection)")
    ax.set_xticks(np.arange(len(scen)), scen)
    ax.set(ylabel="events detected (%)", ylim=(0, 105),
           title="E4: noise bursts and stuck sensors vs mean-based detectors (wear-removed signal)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "E4_noise_stuck.png")
    plt.close(fig)
    txt = [fmt_table(full, ["signal", "scenario", "method", "detection_rate", "placebo_rate", "median_delay",
                            "fa_per_1000"], FMT)]
    g = df.set_index(["scenario", "method"])
    s = [f"A stuck sensor is caught by CUSUM in only {g.loc[('stuck','cusum'),'detection_rate']:.0%} of the events "
         f"(Shewhart {g.loc[('stuck','shewhart'),'detection_rate']:.0%}). The frozen value is just the last noisy reading, "
         "so it acts like a level shift of random size, usually below 1 sigma0; when it freezes close to the mean "
         "nothing in the mean changes at all.",
         f"A noise burst x2 keeps the mean unchanged: CUSUM detects {g.loc[('noise x2','cusum'),'detection_rate']:.0%}, "
         f"EWMA {g.loc[('noise x2','ewma'),'detection_rate']:.0%}. Shewhart does better "
         f"({g.loc[('noise x2','shewhart'),'detection_rate']:.0%}) because wider noise produces extreme single values. "
         "A burst x4 is so wide that every method sees it.",
         "What would catch them: a variance chart (CUSUM or EWMA on z_t squared, or on |z_t - z_(t-1)|) for noise "
         "bursts, and a run-length rule on zero change (for example 5 identical readings in a row) for stuck sensors."]
    return "E4 Noise bursts and stuck sensors: an honest limit", txt, s


def e5(ctx):
    rows = []
    groups = {"4 sensors (s4, s7, s11, s12)": ["s4", "s7", "s11", "s12"], "all 14 sensors": ctx.setup.sensors}
    for sig in SIGNALS:
        for gname, sens in groups.items():
            for d in (0.1, 0.2, 0.3, 0.5, 0.75):
                sc = make_scenario(ctx, injector_cfg(505, mode="joint", level_shift=dict(
                    per_100=0.5, magnitude=[d, d], duration=[30, 60], direction="wear")), inj_sensors=sens,
                    df=signal_df(ctx, sig))
                for r in evaluate(ctx, sc, ["mewma", "cusum", "ewma"], system=True):
                    rows.append(dict(r, signal=sig, group=gname, shift=d))
    full = pd.DataFrame(rows)
    save(full, "E5_joint")
    df = full[full.signal == "wear removed"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for i, gname in enumerate(groups):
        for m in ("mewma", "cusum", "ewma"):
            g = df[(df.group == gname) & (df.method == m)]
            ax[i].plot(g["shift"], g["detection_rate"] * 100, "o-", color=COLORS[m],
                       label=LABELS[m] + ("" if m == "mewma" else " (any sensor)"))
        pl = df[df.group == gname].groupby("shift")["placebo_rate"].max() * 100
        ax[i].plot(pl.index, pl.values, "k:", label="placebo (highest)")
        ax[i].set(title=f"Joint shift on {gname}", xlabel="shift per sensor (sigma0, wear direction)",
                  ylabel="events detected (%)", ylim=(0, 105))
    ax[0].legend(fontsize=8)
    fig.suptitle("E5: the same shift on several sensors at once (wear-removed signal): MEWMA vs 'any sensor' rules")
    fig.tight_layout()
    fig.savefig(FIG / "E5_joint.png")
    plt.close(fig)
    txt = [fmt_table(full, ["signal", "group", "shift", "method", "detection_rate", "placebo_rate", "median_delay",
                            "fa_per_1000"], FMT)]
    g = df.set_index(["group", "shift", "method"])
    a = list(groups)
    s = [f"With a 0.3 sigma0 shift on all 14 sensors MEWMA detects {g.loc[(a[1],0.3,'mewma'),'detection_rate']:.0%} "
         f"(median delay {g.loc[(a[1],0.3,'mewma'),'median_delay']:.0f} cycles), 'any sensor' CUSUM "
         f"{g.loc[(a[1],0.3,'cusum'),'detection_rate']:.0%} (delay {g.loc[(a[1],0.3,'cusum'),'median_delay']:.0f}).",
         f"With only 4 of the 14 sensors shifted by 0.3 sigma0: MEWMA {g.loc[(a[0],0.3,'mewma'),'detection_rate']:.0%}, "
         f"CUSUM {g.loc[(a[0],0.3,'cusum'),'detection_rate']:.0%}. MEWMA adds up evidence over all sensors, so it "
         "gains most when many sensors move a little together.",
         "On the raw signal (see table) wear alarms fill most windows by chance (placebo rate), so the comparison "
         "is only meaningful on the wear-removed signal."]
    return "E5 Joint shift on several sensors: MEWMA vs any-sensor CUSUM", txt, s


def e6(ctx):
    pan = to_panel(ctx.setup.df, ctx.setup.sensors)
    rows = []
    for variant, params in [("iid_est", None), ("real", ctx.real_params())]:
        al = run_all(ctx, pan, METHOD_ORDER, params)
        for m in METHOD_ORDER:
            a = al[m].any(axis=2)
            for e in range(len(pan["units"])):
                L = pan["lengths"][e]
                hits = np.flatnonzero(a[e, :L])
                first = hits[0] + 1 if len(hits) else np.nan
                after = hits[hits >= round(ctx.early_f * L)]
                first_late = after[0] + 1 if len(after) else np.nan
                rows.append(dict(variant=variant, method=m, unit=int(pan["units"][e]), life=int(L),
                                 first_alarm=first, rul_at_first=L - first if len(hits) else np.nan,
                                 first_frac=first / L if len(hits) else np.nan,
                                 rul_at_first_after_early=L - first_late if len(after) else np.nan,
                                 alarms=len(hits)))
    df = pd.DataFrame(rows)
    save(df, "E6_real_wear")
    summ = []
    for (v, m), g in df.groupby(["variant", "method"], sort=False):
        summ.append(dict(variant=v, method=m, engines_alarmed=g.first_alarm.notna().mean(),
                         first_before_early=(g.first_frac < ctx.early_f).mean(),
                         median_rul_first=g.rul_at_first.median(),
                         median_rul_after_early=g.rul_at_first_after_early.median(),
                         p10_rul_after_early=g.rul_at_first_after_early.quantile(0.1),
                         median_alarms=g.alarms.median()))
    sm = pd.DataFrame(summ)
    save(sm, "E6_real_wear_summary")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for i, v in enumerate(["iid_est", "real"]):
        data = [df[(df.variant == v) & (df.method == m)].rul_at_first_after_early.dropna() for m in METHOD_ORDER]
        bp = ax[i].boxplot(data, patch_artist=True, tick_labels=[LABELS[m] for m in METHOD_ORDER])
        for patch, m in zip(bp["boxes"], METHOD_ORDER):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.6)
        ax[i].set(title=f"thresholds: {v}", ylabel="cycles before failure at first alarm")
    fig.suptitle(f"E6: real wear, no injection. Warning time = first alarm after {ctx.early_f:.0%} of life, 100 engines")
    fig.tight_layout()
    fig.savefig(FIG / "E6_real_wear.png")
    plt.close(fig)
    txt = [fmt_table(sm, ["variant", "method", "engines_alarmed", "first_before_early", "median_rul_first",
                          "median_rul_after_early", "p10_rul_after_early", "median_alarms"],
                     {"engines_alarmed": "{:.0%}", "first_before_early": "{:.0%}", "median_rul_first": "{:.0f}",
                      "median_rul_after_early": "{:.0f}", "p10_rul_after_early": "{:.0f}", "median_alarms": "{:.0f}"}),
           "", f"first_before_early = share of engines whose first alarm comes before {ctx.early_f:.0%} of life "
               "(most likely a false alarm). p10 = 10% of the engines get less warning than this."]
    g = sm.set_index(["variant", "method"])
    s = [f"With the default thresholds CUSUM warns a median {g.loc[('iid_est','cusum'),'median_rul_after_early']:.0f} "
         f"cycles before failure (EWMA {g.loc[('iid_est','ewma'),'median_rul_after_early']:.0f}, MEWMA "
         f"{g.loc[('iid_est','mewma'),'median_rul_after_early']:.0f}, Shewhart "
         f"{g.loc[('iid_est','shewhart'),'median_rul_after_early']:.0f}); FD001 engines live 128 to 362 cycles.",
         f"With the 'real' thresholds (early wear treated as noise) the warning comes later: CUSUM "
         f"{g.loc[('real','cusum'),'median_rul_after_early']:.0f} cycles, with fewer early alarms "
         f"({g.loc[('real','cusum'),'first_before_early']:.0%} vs {g.loc[('iid_est','cusum'),'first_before_early']:.0%} of engines "
         "alarm before 30% of life). This is the trade-off a maintenance planner has to choose."]
    return "E6 Real wear, no injection: early warning before failure", txt, s


def e7(ctx):
    """One-at-a-time sensitivity on a standard mixed scenario (config.yaml defaults)."""
    std_cfg = load_yaml(ROOT / "config.yaml")
    std_cfg["seed"] = 707
    sc = make_scenario(ctx, std_cfg, df=ctx.df_nowear)
    p0 = ctx.params
    d = p0["detectors"]
    sweeps = [("shewhart", "k", [d["shewhart"]["k"] * f for f in (0.8, 0.9, 1.0, 1.1, 1.2)]),
              ("sprt", "delta", [0.5, 1.0, 1.5, 2.0]),
              ("sprt", "alpha", [d["sprt"]["alpha"] * f for f in (100, 10, 1, 0.1, 0.01)]),
              ("sprt", "beta", [0.01, 0.05, 0.1, 0.2, 0.4]),
              ("cusum", "kappa", [0.25, 0.5, 0.75, 1.0, 1.5]),
              ("cusum", "h", [d["cusum"]["h"] * f for f in (0.6, 0.8, 1.0, 1.2, 1.4)]),
              ("ewma", "lam", [0.05, 0.1, 0.2, 0.3, 0.45]),
              ("ewma", "L", [d["ewma"]["L"] * f for f in (0.8, 0.9, 1.0, 1.1, 1.2)]),
              ("mewma", "lam", [0.05, 0.1, 0.2, 0.3, 0.45]),
              ("mewma", "h", [d["mewma"]["h"] * f for f in (0.6, 0.8, 1.0, 1.2, 1.4)])]
    rows = []
    kinds = ["level_shift", "drift"]

    def run(method, label, value, params=None, overrides=None, grace=None, B=None, scen=None):
        scen = scen or sc
        al = run_all(ctx, scen["panel"], [method], params, overrides, B=B)[method]
        g = ctx.grace if grace is None else grace
        s = score(scen, al, method, g, ctx.setup.sensors, kinds=kinds)
        sp = score(scen, al, method, 0, ctx.setup.sensors, kinds=["spike"])
        s.pop("_events")
        rows.append(dict(method=method, parameter=label, value=value, detection_rate=s["detection_rate"],
                         mean_delay=s["mean_delay"], fa_per_1000=s["fa_per_1000"],
                         spikes_flagged=sp["detection_rate"],
                         default=_is_default(value, _default(label, method, p0))))

    for m, par, vals in sweeps:
        for v in vals:
            run(m, par, float(v), overrides={m: {par: float(v)}})
    for m in METHOD_ORDER:
        for g in (0, 5, 10, 20, 40):
            run(m, "grace", g, grace=g)
        for rp in ("reset", "continue"):
            run(m, "restart", rp, overrides={m: {"restart": rp}})
        if m != "shewhart":            # clipping below k would switch Shewhart off: not meaningful
            for c in (None, 3.0, 4.0, 6.0):
                run(m, "clip", "none" if c is None else c, overrides={m: {"clip": c}})
    for B in (15, 20, 30, 40, 50):
        # the injector keeps the same window clean; false alarms are counted from the end of it
        cfgB = dict(std_cfg, baseline_cycles=B)
        scB = make_scenario(ctx, cfgB, df=ctx.df_nowear, B=B)
        for m in METHOD_ORDER:
            run(m, "baseline_cycles", B, B=B, scen=scB)
    df = pd.DataFrame(rows)
    save(df, "E7_sensitivity")

    panels = [(m, p) for m, p, _ in sweeps] + [("cusum", "grace"), ("cusum", "clip"), ("cusum", "baseline_cycles"),
                                                ("ewma", "baseline_cycles")]
    n = len(panels)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 3.1 * int(np.ceil(n / cols))))
    for ax, (m, par) in zip(axes.flat, panels):
        g = df[(df.method == m) & (df.parameter == par)]
        x = np.arange(len(g)) if par == "clip" else g["value"].astype(float)
        ax.plot(x, g["detection_rate"] * 100, "o-", color=COLORS[m], label="detected (%)")
        ax.set_ylabel("detected (%)", color=COLORS[m])
        ax.set_ylim(0, 105)
        ax2 = ax.twinx()
        ax2.plot(x, g["fa_per_1000"], "s--", color="crimson", label="false alarms / 1000")
        ax2.set_ylabel("false alarms / 1000 cycles", color="crimson")
        dv = g[g["default"]]
        if len(dv):
            ax.axvline((np.flatnonzero(g["default"].to_numpy())[0]) if par == "clip" else float(dv["value"].iloc[0]),
                       color="k", lw=0.8, ls=":")
        if par == "clip":
            ax.set_xticks(x, g["value"].astype(str))
        if par == "alpha":
            ax.set_xscale("log")
        ax.set_title(f"{LABELS[m]}: {par}")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("E7: one-at-a-time sensitivity (dotted line = default). config.yaml mix, wear-removed signal")
    fig.tight_layout()
    fig.savefig(FIG / "E7_sensitivity.png")
    plt.close(fig)
    df["value_s"] = [f"{v:.3g}" if isinstance(v, float) else str(v) for v in df["value"]]
    df["default_s"] = np.where(df["default"], "yes", "")
    txt = [fmt_table(df, ["method", "parameter", "value_s", "default_s", "detection_rate", "mean_delay",
                          "fa_per_1000", "spikes_flagged"],
                     {"detection_rate": "{:.0%}", "mean_delay": "{:.1f}", "fa_per_1000": "{:.1f}",
                      "spikes_flagged": "{:.0%}"})]
    g = df[(df.method == "cusum") & (df.parameter == "h")]
    gk = df[(df.method == "cusum") & (df.parameter == "kappa")]
    s = [f"Thresholds trade detection against false alarms directly: CUSUM h from {g.value.min():.1f} to "
         f"{g.value.max():.1f} moves detection from {g.detection_rate.iloc[0]:.0%} to {g.detection_rate.iloc[-1]:.0%} "
         f"and false alarms from {g.fa_per_1000.iloc[0]:.1f} to {g.fa_per_1000.iloc[-1]:.1f} per 1000 cycles.",
         f"Changing a shape parameter without recalibrating the threshold mostly changes the false alarm rate "
         f"(CUSUM kappa 0.25: {gk.fa_per_1000.iloc[0]:.1f}, kappa 1.5: {gk.fa_per_1000.iloc[-1]:.1f} per 1000): "
         "always recalibrate h (the app's auto threshold button) after moving kappa or lambda.",
         "The grace window only changes the scoring, not the detector: a short grace counts late alarms of long "
         "events as false alarms."]
    return "E7 Sensitivity of each hyperparameter", txt, s


def _is_default(v, d):
    if isinstance(v, (int, float)) and isinstance(d, (int, float)):
        return bool(np.isclose(v, d))
    return v == d


def _default(label, method, p):
    if label == "grace":
        return int(p["scoring"]["grace"])
    if label == "restart":
        return p.get("restart", "reset")
    if label == "clip":
        return "none" if p.get("clip") is None else float(p["clip"])
    if label == "baseline_cycles":
        return int(p["baseline_cycles"])
    return float(p["detectors"][method][label])


EXPERIMENTS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", default=list(EXPERIMENTS), help="experiments to run (default all)")
    a = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    parts_dir = REP / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for name in a.which:
        t0 = time.time()
        title, txt, sentences = EXPERIMENTS[name](ctx)
        body = f"## {title}\n\n![{name}](figures/{Path(FIG / name).name}" \
               f"{_fig_suffix(name)}.png)\n\n" + " ".join(sentences) + "\n\n" + "\n".join(txt) + "\n"
        (parts_dir / f"{name}.md").write_text(body, encoding="utf-8")
        print(f"{name} done in {time.time() - t0:.0f}s")
        for s_ in sentences:
            print("   ", s_)
    write_results(ctx)


def _fig_suffix(name):
    return {"E1": "_level_shift", "E2": "_drift", "E3": "_spikes", "E4": "_noise_stuck", "E5": "_joint",
            "E6": "_real_wear", "E7": "_sensitivity"}[name]


def write_results(ctx):
    p = ctx.params
    head = [
        "# Stage 3 results: how useful are the detectors?", "",
        "Generated by `python evaluate.py` (settings in `reports/results_settings.yaml`).", "",
        f"Data: C-MAPSS FD001, 100 engines, {len(ctx.setup.sensors)} sensors. Anomalies are injected only in the "
        f"healthy part of life (cycles {ctx.B + 1} to {ctx.clean_f:.0%} of life). All methods use thresholds "
        f"calibrated to the same ARL0 = {p['calibration']['target_arl0']} cycles for the whole system "
        f"(any of the {len(ctx.setup.sensors)} sensors), source `{p['calibration']['source']}`. "
        f"Grace window {ctx.grace} cycles.", "",
        "Columns: `detection_rate` share of events with an alarm in [start, end + grace]; `placebo_rate` the same "
        "on the clean signal (chance level); `median_delay`, `mean_delay` in cycles from the event start "
        "(detected events only); `fa_per_1000` false alarms per 1000 cycles before "
        f"{ctx.clean_f:.0%} of life; `fa_per_1000_early` the same before {ctx.early_f:.0%} of life, where the "
        "engines show no wear yet; `arl0_emp` = 1000 / fa_per_1000.", "",
        "Important: in this data slow wear is already visible between 30% and 60% of life (0.5 to 1 sigma0). "
        "On the raw signal, alarms there are counted as false alarms, but most are early wear detections. Compare "
        "`fa_per_1000_early` (close to the target 2 per 1000) with `fa_per_1000`.", "",
        "Two signals are therefore used. **raw**: the real data as it is. **wear removed**: for each engine and "
        "sensor a smooth cubic trend over the whole life is subtracted (the real noise, rounding and dependence "
        "stay). The wear-removed signal measures the detectors themselves (false alarms back at the target); "
        "the raw signal shows what happens in practice. E1 shows both, E2 to E5 and E7 use wear removed as the "
        "main result (raw rows are in the tables), E6 uses raw data only because there the wear is the target.",
        ""]
    parts = []
    for name in EXPERIMENTS:
        f = REP / "parts" / f"{name}.md"
        if f.exists():
            parts.append(f.read_text(encoding="utf-8"))
    (REP / "results.md").write_text("\n".join(head) + "\n" + "\n".join(parts), encoding="utf-8")
    settings = dict(params=p, sensors=ctx.setup.sensors,
                    sigma0={k: round(v, 6) for k, v in ctx.setup.sigma0_fleet.items()},
                    wear_direction=ctx.setup.wear,
                    seeds=dict(E1=101, E2=202, E3=303, E4=404, E5=505, E6="no injection", E7=707),
                    calibration_file="calibration.json")
    with open(REP / "results_settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False)
    print(f"wrote {REP / 'results.md'}")


if __name__ == "__main__":
    main()
