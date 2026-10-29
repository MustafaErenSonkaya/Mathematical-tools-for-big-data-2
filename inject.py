"""Inject anomalies into C-MAPSS engine data from the command line.

Example:
    python inject.py --data data/train_FD001.txt --config config.yaml \
        --units 1 2 3 --sensors s4 s11 s7 --out out/

Writes:
    out/injected.csv     the data with anomalies + ground-truth columns
    out/events.csv       one row per injected event (where, what, how big)
    out/run_info.yaml    the exact settings, sigma0 and wear directions used
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cmapss import fleet_sigma0, informative_sensors, load_cmapss, wear_directions
from injector import AnomalyInjector, InjectorConfig, inject_frame


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="C-MAPSS file, e.g. data/train_FD001.txt")
    p.add_argument("--config", default="config.yaml", help="injection settings (YAML)")
    p.add_argument("--out", default="out", help="output folder")
    p.add_argument("--units", type=int, nargs="*", help="engine numbers (default: all)")
    p.add_argument("--sensors", nargs="*", help="sensor columns, e.g. s4 s11 (default: all informative)")
    p.add_argument("--seed", type=int, help="override the seed in the config")
    p.add_argument("--sigma", choices=["fleet", "stream"], default="fleet",
                   help="fleet: sigma0 from the healthy start of all engines (stable, default); "
                        "stream: each engine estimates its own from its baseline window")
    p.add_argument("--life-fraction", type=float, default=1.0,
                   help="keep only the first part of each engine's life, e.g. 0.6. "
                        "The end of life has real wear, which detectors will also flag.")
    a = p.parse_args(argv)

    with open(a.config) as f:
        raw = yaml.safe_load(f) or {}
    if a.seed is not None:
        raw["seed"] = a.seed
    cfg = InjectorConfig.from_dict(raw)

    df = load_cmapss(a.data)
    sensors = a.sensors or informative_sensors(df)
    bad = [s for s in sensors if s not in df.columns]
    if bad:
        raise SystemExit(f"unknown sensor(s): {bad}")

    # learned on the full file (all engines, full life) before filtering
    wear = wear_directions(df, sensors)
    sigma0 = fleet_sigma0(df, sensors, cfg.baseline_cycles) if a.sigma == "fleet" else None

    if not 0 < a.life_fraction <= 1:
        raise SystemExit("--life-fraction must be in (0, 1]")
    if a.life_fraction < 1:
        last = df.groupby("unit")["cycle"].transform("max")
        df = df[df["cycle"] <= (last * a.life_fraction).round()]
    if a.units:
        df = df[df["unit"].isin(a.units)]
        if df.empty:
            raise SystemExit("none of those units are in the file")

    short = df.groupby("unit")["cycle"].count()
    short = short[short <= cfg.baseline_cycles]
    if len(short):
        print(f"warning: units {list(short.index)} are not longer than the baseline window; "
              "they get no anomalies")

    inj = AnomalyInjector(sensors, cfg, sigma0=sigma0, wear_dir=wear)
    data, events = inject_frame(df, sensors, inj)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "injected.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    info = dict(data=str(a.data), units=sorted(int(u) for u in df["unit"].unique()),
                sensors=sensors, sigma_source=a.sigma, life_fraction=a.life_fraction,
                sigma0={s: round(v, 6) for s, v in (sigma0 or {}).items()} or "per stream",
                wear_direction=wear, config=cfg.to_dict())
    with open(out / "run_info.yaml", "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)

    n = len(data)
    print(f"{data['unit'].nunique()} engines, {n} cycles, {len(sensors)} sensors")
    if events.empty:
        print("no events injected (check per_100 / enabled)")
    else:
        per = events.groupby("kind").agg(events=("event_id", "nunique"),
                                         mean_size_sigma=("magnitude_sigma", lambda v: v.abs().mean()),
                                         mean_length=("length", "mean"))
        print(per.round(2).to_string())
        print(f"labelled samples: {100 * data[[f'{s}_label' for s in sensors]].values.mean():.1f}% "
              f"of sensor-cycles")
    print(f"wrote {out/'injected.csv'}, {out/'events.csv'}, {out/'run_info.yaml'}")


if __name__ == "__main__":
    main()
