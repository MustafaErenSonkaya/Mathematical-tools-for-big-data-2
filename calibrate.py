"""Calibrate every detector to the same in-control ARL0.

Fair comparison rule: each method gets the alarm threshold (Shewhart k,
SPRT alpha, CUSUM h, EWMA L, MEWMA h) that gives the SAME average run length
to a false alarm, ARL0, for the whole system ("any of the p sensors alarms";
MEWMA has one statistic anyway).

ARL0 is measured as   monitored in-control cycles / number of alarms,
restarting the detector after each alarm (each restart = a new run).

Four calibration sources, same structure (same number of engines, same
monitored lengths, same 30 cycle baseline):

  iid      simulated iid N(0,1) noise, mu0 and sigma0 known (textbook case)
  iid_est  the same noise, but mu0 estimated from the 30 baseline cycles of
           each simulated engine (like the real system does)
  real     the real clean data: cycles 31 .. 60% of life of all 100 engines,
           no injection, mu0 per engine, sigma0 and Sigma pooled over the fleet
  real_early  the same, but only up to 30% of life (before wear starts to show)

iid -> iid_est shows the cost of estimating mu0 from only 30 points;
iid_est -> real_early shows the effect of real noise (dependence, rounding,
non-Gaussian tails); real_early -> real shows the effect of slow wear
("restarts and dependence need calibration"). A sweep over the end of the
"healthy" window (30% .. 60% of life) is written to the report as well.

Writes calibration.json (read by params.yaml "auto" and the app's
"auto threshold" button), reports/calibration.md, and
reports/figures/calibration_curves.png.

    python calibrate.py              # full grid (a few minutes)
    python calibrate.py --quick      # default shapes only, about 30 s. Writes
                                     # calibration_quick.json and *_quick reports, so
                                     # the full calibration.json is never replaced
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from detectors import SHAPE_PARAM, THRESHOLD_PARAM
from pipeline import (LABELS, METHOD_ORDER, ROOT, baseline_stats, clean_mask, load_params,
                      load_setup, run_method, system_alarms, to_panel)

SHAPE_GRID = {
    "shewhart": [None],
    "sprt": [0.5, 1.0, 1.5, 2.0, 3.0],
    "cusum": [0.25, 0.5, 0.75, 1.0, 1.5],
    "ewma": [0.05, 0.1, 0.2, 0.3, 0.45],
    "mewma": [0.05, 0.1, 0.2, 0.3, 0.45],
}
SPRT_BETAS = [0.05, 0.1, 0.2]
# starting bracket for the threshold variable u (SPRT: u = log A)
BRACKET = {"shewhart": (2.5, 5.5), "sprt": (2.0, 12.0), "cusum": (2.0, 12.0),
           "ewma": (2.0, 4.0), "mewma": (15.0, 50.0)}
SOURCES = ["iid", "iid_est", "real_early", "real"]
SWEEP_FRACTIONS = [0.3, 0.4, 0.5, 0.6]


# --------------------------------------------------------------------------
# the data each source is measured on
# --------------------------------------------------------------------------
class Source:
    def __init__(self, name, X, mu0, sigma0, mask, cov, B):
        self.name, self.X, self.mu0, self.sigma0, self.mask, self.cov, self.B = \
            name, X, mu0, sigma0, mask, cov, B
        self.cycles = int(mask.sum())

    def arl0(self, method, kw):
        al = system_alarms(run_method(method, kw, self.X, self.mu0, self.sigma0, self.B, cov=self.cov))
        n = int((al & self.mask).sum())
        return (self.cycles / n if n else math.inf), n


def real_source(setup, life_fraction, name="real"):
    B = setup.baseline_cycles
    pan = to_panel(setup.df, setup.sensors)
    mask = clean_mask(pan, B, life_fraction)
    T = int(mask.any(axis=0).nonzero()[0].max()) + 1
    X, mask = pan["X"][:, :T], mask[:, :T]
    mu0, sig = baseline_stats(X, B, setup.sigma0_vec, "fleet")
    return Source(name, X, mu0, sig, mask, setup.cov, B)


def build_sources(setup, life_fraction, copies, seed, early_fraction=0.3):
    B = setup.baseline_cycles
    pan = to_panel(setup.df, setup.sensors)
    mask = clean_mask(pan, B, life_fraction)
    T = int(mask.any(axis=0).nonzero()[0].max()) + 1           # nothing to count after this step
    X = pan["X"][:, :T]
    mask = mask[:, :T]
    mu0, sig = baseline_stats(X, B, setup.sigma0_vec, "fleet")
    real = Source("real", X, mu0, sig, mask, setup.cov, B)

    # simulated engines with the same monitored lengths as the real ones
    rng = np.random.default_rng(seed)
    E, _, p = X.shape
    m_sim = np.tile(mask, (copies, 1))
    noise = rng.standard_normal((E * copies, T, p))
    noise[~np.tile(np.arange(T)[None, :] < (mask.sum(1)[:, None] + B), (copies, 1))] = np.nan
    zero = np.zeros((E * copies, p))
    iid = Source("iid", noise, zero, np.ones(p), m_sim, np.eye(p), B)
    iid_est = Source("iid_est", noise, noise[:, :B].mean(axis=1), np.ones(p), m_sim, np.eye(p), B)
    return {"iid": iid, "iid_est": iid_est, "real": real,
            "real_early": real_source(setup, early_fraction, "real_early")}


# --------------------------------------------------------------------------
# threshold search
# --------------------------------------------------------------------------
def kw_for(method, shape, u, beta=None, base=None):
    kw = dict(base or {})
    sp, tp = SHAPE_PARAM[method], THRESHOLD_PARAM[method]
    if sp:
        kw[sp] = shape
    if method == "sprt":
        kw["beta"] = beta
        kw["alpha"] = (1 - beta) / math.exp(u)     # u = log A
    else:
        kw[tp] = u
    return kw


def u_to_threshold(method, u, beta=None):
    return (1 - beta) / math.exp(u) if method == "sprt" else u


def calibrate_one(src, method, shape, targets, beta=None, base=None, n_grid=12):
    """Find u (threshold variable) giving each target ARL0 on this source."""
    f = lambda u: src.arl0(method, kw_for(method, shape, u, beta, base))[0]
    lo, hi = BRACKET[method]
    lo_t, hi_t = min(targets) * 0.7, max(targets) * 1.4
    a_lo = f(lo)
    for _ in range(12):
        if a_lo <= lo_t:
            break
        lo = lo * 0.6 if lo > 0 else lo - 1
        a_lo = f(lo)
    a_hi = f(hi)
    for _ in range(12):
        if a_hi >= hi_t:
            break
        hi = hi * 1.4
        a_hi = f(hi)
    us = np.linspace(lo, hi, n_grid)
    arls = np.array([a_lo] + [f(u) for u in us[1:-1]] + [a_hi])
    cap = src.cycles * 2.0
    la = np.log(np.minimum(arls, cap))
    la = np.maximum.accumulate(la)                    # ARL0 must grow with the threshold
    out = {}
    for tg in targets:
        u = float(np.interp(math.log(tg), la + np.arange(len(la)) * 1e-9, us))
        # two secant refinements in log-ARL space
        for _ in range(2):
            a = f(u)
            if not math.isfinite(a):
                break
            i = int(np.clip(np.searchsorted(la, math.log(a)), 1, len(us) - 1))
            slope = (la[i] - la[i - 1]) / (us[i] - us[i - 1]) if us[i] != us[i - 1] else 0
            if slope <= 0:
                break
            u = float(np.clip(u + (math.log(tg) - math.log(a)) / slope, lo, hi))
        a, n = src.arl0(method, kw_for(method, shape, u, beta, base))
        out[str(tg)] = dict(u=u, threshold=u_to_threshold(method, u, beta), arl0=a, alarms=n)
    curve = dict(u=us.tolist(), arl0=[float(min(x, cap)) for x in arls])
    return out, curve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="only the default shape of each method")
    ap.add_argument("--copies", type=int, default=20,
                    help="simulated fleets per iid source (20 = the saved calibration.json)")
    ap.add_argument("--out", help="output file (default: calibration.json, or calibration_quick.json "
                                  "with --quick)")
    a = ap.parse_args(argv)
    # a quick run only knows the default kappa / lambda / delta; writing it over the full
    # calibration.json would silently give wrong thresholds for every other setting
    a.out = a.out or str(ROOT / ("calibration_quick.json" if a.quick else "calibration.json"))

    params = load_params(resolve=False)
    c = params["calibration"]
    targets = sorted({int(x) for x in c["targets"]} | {int(c["target_arl0"])})
    setup = load_setup(params)
    srcs = build_sources(setup, float(c["clean_life_fraction"]), a.copies, int(c["seed"]),
                         float(c.get("early_life_fraction", 0.3)))
    base = dict(restart=params.get("restart", "reset"), clip=params.get("clip"))
    print(f"{len(setup.sensors)} sensors, rule = any sensor alarms; monitored in-control cycles: "
          + ", ".join(f"{k} {v.cycles}" for k, v in srcs.items()))

    defaults = params["detectors"]
    results = {m: [] for m in METHOD_ORDER}
    curves = {}
    t0 = time.time()
    for m in METHOD_ORDER:
        sp = SHAPE_PARAM[m]
        shapes = [defaults[m][sp]] if (a.quick and sp) else SHAPE_GRID[m]
        betas = ([defaults["sprt"]["beta"]] if a.quick else SPRT_BETAS) if m == "sprt" else [None]
        for shape in shapes:
            for beta in betas:
                for sname in SOURCES:
                    res, curve = calibrate_one(srcs[sname], m, shape, targets, beta, base)
                    row = dict(source=sname, thresholds={k: v["threshold"] for k, v in res.items()},
                               check_arl0={k: round(v["arl0"], 1) for k, v in res.items()},
                               check_alarms={k: v["alarms"] for k, v in res.items()})
                    if sp:
                        row[sp] = shape
                    if beta is not None:
                        row["beta"] = beta
                        row["log_a"] = {k: v["u"] for k, v in res.items()}
                    results[m].append(row)
                    is_default = (sp is None or shape == defaults[m][sp]) and \
                                 (beta is None or beta == defaults["sprt"]["beta"])
                    if is_default:
                        curves[(m, sname)] = curve
                    print(f"  {m:8s} {sp or '':5s}={shape!s:5s} {'beta=' + str(beta) if beta else '':9s} "
                          f"{sname:8s} thr@{c['target_arl0']} = {res[str(c['target_arl0'])]['threshold']:.5g}"
                          f"  (check ARL0 {res[str(c['target_arl0'])]['arl0']:.0f})   [{time.time() - t0:.0f}s]")

    # cross check: thresholds from iid used on the real data
    cross = {}
    tg = str(c["target_arl0"])
    for m in METHOD_ORDER:
        sp = SHAPE_PARAM[m]
        row = next(r for r in results[m] if r["source"] == "iid" and (sp is None or r[sp] == defaults[m][sp])
                   and (m != "sprt" or r["beta"] == defaults["sprt"]["beta"]))
        kw = dict(base, **{k: v for k, v in defaults[m].items() if v != "auto"})
        kw[THRESHOLD_PARAM[m]] = row["thresholds"][tg]
        cross[m] = round(srcs["real"].arl0(m, kw)[0], 1)

    # how the real-data threshold grows with the end of the "healthy" window
    sweep = {}
    for f in SWEEP_FRACTIONS:
        src = real_source(setup, f)
        sweep[str(f)] = {"cycles": src.cycles}
        for m in METHOD_ORDER:
            sp = SHAPE_PARAM[m]
            res, _ = calibrate_one(src, m, defaults[m][sp] if sp else None, [int(tg)],
                                   defaults["sprt"]["beta"] if m == "sprt" else None, base, n_grid=8)
            sweep[str(f)][m] = res[tg]["threshold"]
        print(f"  sweep life fraction {f}: " + ", ".join(f"{m} {sweep[str(f)][m]:.4g}" for m in METHOD_ORDER))

    cal = dict(target_arl0=int(c["target_arl0"]), targets=targets, default_source=c["source"],
               sensors=setup.sensors, p=len(setup.sensors), rule="any sensor alarms (MEWMA: T2 > h)",
               baseline_cycles=setup.baseline_cycles, clean_life_fraction=c["clean_life_fraction"],
               monitored_cycles={k: v.cycles for k, v in srcs.items()}, sim_copies=a.copies,
               seed=c["seed"], restart=base["restart"], clip=base["clip"],
               early_life_fraction=c.get("early_life_fraction", 0.3),
               iid_threshold_on_real_arl0=cross, life_fraction_sweep=sweep, results=results,
               note="thresholds are in sigma0 units (k, h, L), T2 units (MEWMA h) or alpha (SPRT)")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=1)
    print(f"wrote {a.out}")
    write_report(cal, curves, defaults, suffix="_quick" if a.quick else "")


def write_report(cal, curves, defaults, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = ROOT / "reports"
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    tg = str(cal["target_arl0"])
    lines = [f"# Calibration (target ARL0 = {tg} cycles, {cal['p']} sensors, rule: {cal['rule']})", "",
             f"Monitored in-control cycles: real (to {cal['clean_life_fraction']} of life) "
             f"{cal['monitored_cycles']['real']}, real_early (to {cal['early_life_fraction']}) "
             f"{cal['monitored_cycles']['real_early']}, simulated {cal['monitored_cycles']['iid']} "
             f"({cal['sim_copies']} simulated fleets with the same engine lengths).", "",
             f"Default source used by params.yaml \"auto\": **{cal['default_source']}**.", "",
             "| method | shape | threshold | iid | iid_est | real_early | real | real ARL0 with the iid threshold |",
             "|---|---|---|---|---|---|---|---|"]
    for m in METHOD_ORDER:
        sp = SHAPE_PARAM[m]
        vals = {}
        for r in cal["results"][m]:
            if (sp is None or r[sp] == defaults[m][sp]) and (m != "sprt" or r["beta"] == defaults["sprt"]["beta"]):
                vals[r["source"]] = (r["thresholds"][tg], r["check_arl0"][tg])
        fmt = (lambda v: f"{v:.2e}") if m == "sprt" else (lambda v: f"{v:.3f}")
        shape = f"{sp} = {defaults[m][sp]}" if sp else "-"
        if m == "sprt":
            shape += f", beta = {defaults['sprt']['beta']}"
        lines.append(f"| {LABELS[m]} | {shape} | {THRESHOLD_PARAM[m]} | "
                     + " | ".join(f"{fmt(vals[s][0])} (ARL0 {vals[s][1]:.0f})" for s in SOURCES)
                     + f" | {cal['iid_threshold_on_real_arl0'][m]:.0f} |")
    lines += ["", "The value in brackets is the ARL0 measured again with that threshold (a check). "
              "real_early allows only a few alarms in total, so its check is noisy.",
              "The last column uses the textbook (iid) threshold on the real data up to "
              f"{cal['clean_life_fraction']} of life: how wrong you would be without calibrating on the real signal.",
              "", "## Where does the difference come from? Threshold on real data vs end of the healthy window", "",
              "| healthy window ends at | cycles | " + " | ".join(LABELS[m] for m in METHOD_ORDER) + " |",
              "|---|---|" + "---|" * len(METHOD_ORDER)]
    for f, row in cal["life_fraction_sweep"].items():
        lines.append(f"| {float(f):.0%} of life | {row['cycles']} | "
                     + " | ".join((f"{row[m]:.2e}" if m == "sprt" else f"{row[m]:.2f}") for m in METHOD_ORDER) + " |")
    (rep / f"calibration{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    styles = {"iid": ("#1f77b4", "-"), "iid_est": ("#ff7f0e", "--"), "real_early": ("#9467bd", "-."),
              "real": ("#2ca02c", "-")}
    for ax, m in zip(axes, METHOD_ORDER):
        for s in SOURCES:
            cv = curves.get((m, s))
            if not cv:
                continue
            x = np.array(cv["u"])
            xl = THRESHOLD_PARAM[m]
            if m == "sprt":
                xl = "log A = log((1 - beta) / alpha)"
            col, ls = styles[s]
            ax.semilogy(x, cv["arl0"], ls, color=col, marker="o", ms=3, label=s)
            ax.set_xlabel(xl)
        ax.axhline(cal["target_arl0"], color="k", lw=0.8, ls=":", label=f"target {tg}")
        sp = SHAPE_PARAM[m]
        ax.set_title(f"{LABELS[m]}" + (f" ({sp} = {defaults[m][sp]})" if sp else ""))
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("in-control ARL0 (cycles, log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Calibration: ARL0 of the whole system ({cal['p']} sensors) against the threshold")
    fig.tight_layout()
    fig.savefig(rep / "figures" / f"calibration_curves{suffix}.png", dpi=120)
    print(f"wrote {rep / f'calibration{suffix}.md'} and the figure")


if __name__ == "__main__":
    main()
