"""Plot one engine: clean signal vs injected signal, with every event marked.

    python plot_injection.py --dir out --unit 1 --sensors s4 s11 s7 --png out/unit1.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from cmapss import sensor_label

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
INJECTED = "#2a78d6"
CLEAN = "#a3a29b"
KIND_COLOR = {"level_shift": "#eb6834", "drift": "#1baf7a", "noise_burst": "#eda100",
              "spike": "#e87ba4", "stuck": "#6250d6"}
KIND_NAME = {"level_shift": "level shift", "drift": "drift", "noise_burst": "noise burst",
             "spike": "spike", "stuck": "stuck sensor"}


def plot_unit(data: pd.DataFrame, events: pd.DataFrame, unit: int, sensors: list[str],
              baseline: int, title: str | None = None):
    d = data[data.unit == unit]
    ev = events[events.unit == unit]
    fig, axes = plt.subplots(len(sensors), 1, figsize=(11, 2.6 * len(sensors) + 0.6),
                             sharex=True, squeeze=False)
    used = set()
    for ax, s in zip(axes[:, 0], sensors):
        ax.axvspan(d.cycle.min() - 0.5, baseline + 0.5, color=GRID, alpha=0.6, lw=0)
        ax.text(d.cycle.min() + 1, 1.0, "baseline (clean)", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=8, color=MUTED)
        for _, e in ev[(ev.sensor == s) & (ev.kind != "spike")].iterrows():
            c = KIND_COLOR[e.kind]
            ax.axvspan(e.start - 0.5, e.end + 0.5, color=c, alpha=0.16, lw=0)
            label = f"{KIND_NAME[e.kind]} {e.magnitude_sigma:+.1f}σ" if e.kind not in ("stuck", "noise_burst") \
                else (f"noise ×{e.magnitude_sigma:.1f}" if e.kind == "noise_burst" else "stuck")
            ax.text((e.start + e.end) / 2, 1.0, label, transform=ax.get_xaxis_transform(),
                    va="top", ha="center", fontsize=8, color=INK)
            used.add(e.kind)
        ax.plot(d.cycle, d[f"{s}_clean"], color=CLEAN, lw=1.0, zorder=2)
        ax.plot(d.cycle, d[s], color=INJECTED, lw=1.4, zorder=3)
        sp = d[d[f"{s}_kind"] == "spike"]
        if len(sp):
            ax.scatter(sp.cycle, sp[s], s=70, facecolor="none", edgecolor=KIND_COLOR["spike"],
                       linewidth=2, zorder=4)
            used.add("spike")
        ax.set_ylabel(sensor_label(s), fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.margins(x=0.01, y=0.18)
    axes[-1, 0].set_xlabel("cycle (one flight per cycle)", fontsize=9, color=INK)
    handles = [Line2D([], [], color=CLEAN, lw=1.0, label="clean (original data)"),
               Line2D([], [], color=INJECTED, lw=1.4, label="with injected anomalies")]
    for k in ("spike", "level_shift", "drift", "noise_burst", "stuck"):
        if k in used:
            handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none",
                                  markeredgecolor=KIND_COLOR[k], markeredgewidth=2,
                                  markersize=8, label="spike") if k == "spike"
                           else Patch(color=KIND_COLOR[k], alpha=0.35, label=KIND_NAME[k]))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title or f"Engine {unit}: injected anomalies vs clean signal", fontsize=12,
                 color=INK, y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="out", help="folder written by inject.py")
    p.add_argument("--unit", type=int, default=1)
    p.add_argument("--sensors", nargs="*")
    p.add_argument("--baseline", type=int, help="default: read from run_info.yaml")
    p.add_argument("--png", help="output image (default: <dir>/unit<N>.png)")
    a = p.parse_args(argv)
    import yaml
    folder = Path(a.dir)
    info = yaml.safe_load(open(folder / "run_info.yaml"))
    data = pd.read_csv(folder / "injected.csv")
    events = pd.read_csv(folder / "events.csv")
    sensors = a.sensors or info["sensors"][:3]
    baseline = a.baseline or info["config"]["baseline_cycles"]
    fig = plot_unit(data, events, a.unit, sensors, baseline)
    out = a.png or folder / f"unit{a.unit}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
