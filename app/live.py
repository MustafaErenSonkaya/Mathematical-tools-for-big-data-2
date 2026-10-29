"""The live system: replays C-MAPSS one cycle at a time through the injector
and all detectors, keeps score, and publishes everything as Prometheus metrics.

One background thread calls step() every 1/speed seconds. All state changes
(from the web page / REST API) take the same lock, so a slider never changes
a detector halfway through a cycle.
"""
from __future__ import annotations

import collections
import copy
import math
import threading
import time

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from detectors import METHODS, SHAPE_PARAM, THRESHOLD_PARAM, make
from injector import ALL_KINDS, AnomalyInjector, InjectorConfig
from pipeline import (ROOT, calibrated_shape_range, calibrated_threshold, detector_kwargs, load_calibration,
                      load_params, load_setup, load_yaml)

SCRAPE_INTERVAL_S = 1.0           # must match prometheus/prometheus.yml
MAX_SPEED = 1.0 / SCRAPE_INTERVAL_S
HOLD_S = 2.0                      # a 1-cycle flag stays visible this long, so a scrape cannot miss it
UNIVARIATE = ["shewhart", "sprt", "cusum", "ewma"]

PRESETS = {
    "calm": {k: {"enabled": False} for k in ALL_KINDS},
    "cusum_demo": dict({k: {"enabled": False} for k in ALL_KINDS},
                       level_shift={"enabled": True, "per_100": 1.5, "magnitude": [1.0, 2.0],
                                    "duration": [30, 50], "direction": "both"}),
    "ewma_demo": dict({k: {"enabled": False} for k in ALL_KINDS},
                      drift={"enabled": True, "per_100": 1.0, "magnitude": [2.0, 4.0],
                             "duration": [40, 70], "hold": [10, 20], "direction": "both"}),
    "spikes_only": dict({k: {"enabled": False} for k in ALL_KINDS},
                        spike={"enabled": True, "per_100": 5.0, "magnitude": [5.0, 9.0], "direction": "both"}),
}


class Metrics:
    def __init__(self):
        r = self.registry = CollectorRegistry()
        self.engine_unit = Gauge("engine_unit", "Engine (unit) being replayed", registry=r)
        self.engine_cycle = Gauge("engine_cycle", "Current cycle of the replayed engine", registry=r)
        self.engine_life = Gauge("engine_life_cycles", "Total cycles of this engine (run to failure)", registry=r)
        self.life_fraction = Gauge("engine_life_fraction", "Cycle / total life", registry=r)
        self.in_baseline = Gauge("engine_in_baseline", "1 while the detectors are still learning mu0", registry=r)
        self.playing = Gauge("replay_playing", "1 when the replay is running", registry=r)
        self.speed = Gauge("replay_speed_cycles_per_second", "Replay speed", registry=r)
        self.sensor_value = Gauge("sensor_value", "Sensor value in its own unit", ["sensor", "series"], registry=r)
        self.sensor_z = Gauge("sensor_zscore", "(value - mu0) / sigma0", ["sensor"], registry=r)
        self.stat = Gauge("detector_statistic", "Detector statistic (sigma0 units; MEWMA: T2)",
                          ["method", "sensor", "side"], registry=r)
        self.thr = Gauge("detector_threshold", "Alarm limit in the same units as the statistic",
                         ["method", "sensor", "side"], registry=r)
        self.alarm = Gauge("detector_alarm", "1 when the detector alarms (held 2 s)", ["method", "sensor"], registry=r)
        self.event = Gauge("injected_event_active", "1 while an injected event is active (spikes held 2 s)",
                           ["sensor", "kind"], registry=r)
        self.alarms_total = Counter("detector_alarms", "Alarms by outcome: true (inside an injected event + grace), "
                                    "false (healthy zone, no event), wear (late life, no event)",
                                    ["method", "outcome"], registry=r)
        self.events_total = Counter("injected_events", "Injected events (per sensor)", ["kind"], registry=r)
        self.detected_total = Counter("detected_events", "Injected events detected (per sensor)",
                                      ["method", "kind"], registry=r)
        self.delay = Histogram("detection_delay_cycles", "First alarm cycle - event start",
                               ["method", "kind"], buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89), registry=r)
        self.engines_done = Counter("engines_replayed", "Engines replayed to the end", registry=r)


class LiveSystem:
    def __init__(self):
        self.lock = threading.RLock()
        self.params = load_params()
        self.raw_params = load_params(resolve=False)
        self.cal = load_calibration()
        self.setup = load_setup(self.params)
        self.S = self.setup.sensors
        self.B = self.setup.baseline_cycles
        self.s0 = self.setup.sigma0_vec
        self.units = sorted(int(u) for u in self.setup.df["unit"].unique())
        sc = self.params["scenario"]
        self.unit = int(sc.get("engine", 1))
        self.loop = bool(sc.get("loop_engines", True))
        self.speed = min(float(sc.get("replay_speed", 1.0)), MAX_SPEED)
        self.playing = True
        self.grace = int(self.params["scoring"]["grace"])
        self.clean_f = float(self.params["scoring"]["clean_life_fraction"])
        self.target = int(self.params["calibration"]["target_arl0"])
        self.cal_source = self.params["calibration"]["source"]
        self.det_params = {m: detector_kwargs(self.params, m) for m in METHODS}
        self.inj_cfg = InjectorConfig.from_dict(load_yaml(ROOT / sc.get("injector_config", "config.yaml")))
        self.m = Metrics()
        self.recent = collections.deque(maxlen=40)
        self._held = {}
        self._stop = threading.Event()
        self.load_engine(self.unit)

    # ------------------------------------------------------------------ engine
    def load_engine(self, unit):
        with self.lock:
            if unit not in self.units:
                raise ValueError(f"engine must be one of 1..{max(self.units)}")
            self.unit = unit
            g = self.setup.df[self.setup.df["unit"] == unit].sort_values("cycle")
            self.rows = g[self.S].to_dict("records")
            self.life = len(self.rows)
            self.idx = 0
            self._done = False
            self.inj = AnomalyInjector(self.S, InjectorConfig.from_dict(self.inj_cfg.to_dict()),
                                       sigma0=self.setup.sigma0_fleet, wear_dir=self.setup.wear)
            self.inj.reset(stream_id=unit)
            self.base = []
            self.dets = None
            self.mu0 = None
            self._detected = set()
            self._counted = set()
            for s in self.S:
                for k in ALL_KINDS:
                    self.m.event.labels(s, k).set(0)
                for m in METHODS:
                    self.m.alarm.labels(m, s).set(0)
            self._nan_stats()
            self.m.engine_unit.set(unit)
            self.m.engine_life.set(self.life)
            self.m.engine_cycle.set(0)
            self.m.life_fraction.set(0)
            self.m.in_baseline.set(1)

    def _make_detectors(self):
        self.mu0 = np.mean(self.base, axis=0)
        self.dets = {}
        for m in METHODS:
            kw = dict(self.det_params[m])
            if m == "mewma":
                self.dets[m] = make(m, mu0=self.mu0, sigma0=self.s0, cov=self.setup.cov, **kw)
            else:
                self.dets[m] = make(m, mu0=self.mu0, sigma0=self.s0, **kw)

    def _nan_stats(self):
        nan = float("nan")
        sides = {"shewhart": ("z",), "ewma": ("z",), "cusum": ("up", "down"), "sprt": ("up", "down")}
        for s in self.S:
            for m, ss in sides.items():
                for side in ss:
                    self.m.stat.labels(m, s, side).set(nan)
            self.m.sensor_z.labels(s).set(nan)
        self.m.stat.labels("mewma", "all", "t2").set(nan)

    # ------------------------------------------------------------------ main step
    def step(self):
        with self.lock:
            if self.idx >= self.life:
                if not self._done:
                    self._done = True
                    self.m.engines_done.inc()
                if self.loop:
                    nxt = self.units[(self.units.index(self.unit) + 1) % len(self.units)]
                    self.load_engine(nxt)
                else:
                    self.playing = False
                return
            row = self.rows[self.idx]
            self.idx += 1
            r = self.inj.step(row)
            t = r["t"]
            now = time.monotonic()
            x = np.array([r["values"][s] for s in self.S])
            m = self.m
            m.engine_cycle.set(t)
            m.life_fraction.set(t / self.life)
            m.in_baseline.set(1 if t <= self.B else 0)
            for j, s in enumerate(self.S):
                m.sensor_value.labels(s, "raw").set(x[j])
                m.sensor_value.labels(s, "clean").set(float(row[s]))
            self._update_events(r, t, now)
            if t <= self.B:
                self.base.append(x)
                if t == self.B:
                    self._make_detectors()
                self._refresh_held(now)
                return
            z = (x - self.mu0) / self.s0
            for j, s in enumerate(self.S):
                m.sensor_z.labels(s).set(z[j])
            for meth, det in self.dets.items():
                out = det.update(x, t=t)
                self._publish(meth, out, t, now)
            self._refresh_held(now)

    def _update_events(self, r, t, now):
        for s in self.S:
            k = r["kinds"][s]
            if k:
                self._held[("event", s, k)] = now + HOLD_S
                key = (self.unit, r["event_ids"][s], s)
                if key not in self._counted:
                    self._counted.add(key)
                    self.m.events_total.labels(k).inc()

    def _refresh_held(self, now):
        for key, until in list(self._held.items()):
            on = now < until
            if key[0] == "event":
                self.m.event.labels(key[1], key[2]).set(1 if on else 0)
            else:
                self.m.alarm.labels(key[1], key[2]).set(1 if on else 0)
            if not on:
                del self._held[key]

    def _publish(self, meth, out, t, now):
        m, S = self.m, self.S
        if meth == "mewma":
            m.stat.labels(meth, "all", "t2").set(out["statistic"])
            m.thr.labels(meth, "all", "h").set(out["threshold"])
            alarms = [S[out["top_sensor"]]] if out["alarm"] else []
        else:
            if meth in ("shewhart", "ewma"):
                stat = {"z": out["statistic"]}
                thr = {"up": out["threshold_up"], "down": out["threshold_down"]}
            elif meth == "cusum":
                stat = {"up": out["statistic_up"], "down": out["statistic_down"]}
                thr = {"h": out["threshold"]}
            else:
                stat = {"up": out["statistic_up"], "down": out["statistic_down"]}
                thr = {"upper": out["threshold_upper"], "lower": out["threshold_lower"]}
            for j, s in enumerate(S):
                for side, v in stat.items():
                    m.stat.labels(meth, s, side).set(float(v[j]))
                for side, v in thr.items():
                    m.thr.labels(meth, s, side).set(float(v[j]))
            alarms = [S[j] for j in np.flatnonzero(out["alarm"])]
        if alarms:
            self._held[("alarm", meth, "any")] = now + HOLD_S
        for s in alarms:
            self._held[("alarm", meth, s)] = now + HOLD_S
            outcome, ev = self._judge(s, t)
            m.alarms_total.labels(meth, outcome).inc()
            if ev is not None:
                key = (self.unit, ev.event_id, s, meth)
                if key not in self._detected:
                    self._detected.add(key)
                    m.delay.labels(meth, ev.kind).observe(t - ev.start)
                    m.detected_total.labels(meth, ev.kind).inc()
            self.recent.appendleft(dict(engine=self.unit, cycle=t, method=meth, sensor=s, outcome=outcome,
                                        kind=ev.kind if ev else ""))

    def _judge(self, s, t):
        for ev in self.inj.events:
            end = ev.end if ev.end is not None else math.inf
            if ev.sensor == s and ev.start <= t <= end + self.grace:
                return "true", ev
        return ("wear" if t / self.life > self.clean_f else "false"), None

    # ------------------------------------------------------------------ loop
    def run_forever(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            if self.playing:
                try:
                    self.step()
                except Exception as e:          # keep the replay alive, show the error on the page
                    self.recent.appendleft(dict(engine=self.unit, cycle=self.idx, method="error", sensor="",
                                                outcome=str(e), kind=""))
            else:
                with self.lock:                 # paused: let the 2 s alarm / event flags switch off
                    self._refresh_held(time.monotonic())
            self.m.playing.set(1 if self.playing else 0)
            self.m.speed.set(self.speed)
            self._stop.wait(max(0.0, 1.0 / self.speed - (time.monotonic() - t0)))

    def start(self):
        th = threading.Thread(target=self.run_forever, daemon=True, name="replay")
        th.start()
        return th

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------ controls
    def config(self):
        with self.lock:
            return dict(
                replay=dict(engine=self.unit, playing=self.playing, speed=self.speed, max_speed=MAX_SPEED,
                            loop=self.loop, cycle=self.idx, life=self.life),
                injector=self.inj.cfg.to_dict(),
                detectors={m: {k: v for k, v in p.items()} for m, p in self.det_params.items()},
                threshold_param=THRESHOLD_PARAM, shape_param=SHAPE_PARAM,
                calibration=dict(target_arl0=self.target, source=self.cal_source,
                                 targets=(self.cal or {}).get("targets", []),
                                 sources=["iid", "iid_est", "real_early", "real"]),
                scoring=dict(grace=self.grace, clean_life_fraction=self.clean_f),
                sensors=self.S, units=self.units, presets=list(PRESETS), baseline_cycles=self.B)

    def set_injector(self, changes: dict):
        """Live change of injector settings. seed: rebuilds the injector and restarts the engine."""
        with self.lock:
            changes = copy.deepcopy(changes)
            new = self.inj.cfg.to_dict()
            kinds = changes.pop("kinds", None) or {}
            if not isinstance(kinds, dict):
                raise ValueError("injector.kinds must be an object like {\"spike\": {\"per_100\": 2}}")
            for k, spec in kinds.items():
                if k not in new["kinds"]:
                    raise ValueError(f"unknown anomaly kind '{k}', use one of {list(ALL_KINDS)}")
                if not isinstance(spec, dict):
                    raise ValueError(f"injector.kinds.{k} must be an object of settings, got {spec!r}")
                new["kinds"][k].update(spec)
            bad = set(changes) - {"seed", "mode", "min_gap"}
            if bad:
                raise ValueError(f"cannot change {sorted(bad)} live; allowed: kinds, seed, mode, min_gap")
            new.update(changes)
            cfg = InjectorConfig.from_dict(new)          # validates, raises ValueError
            self.inj_cfg = cfg
            if "seed" in changes:
                self.load_engine(self.unit)
            else:
                live = {k: v for k, v in cfg.to_dict().items() if k in ("kinds", "mode", "min_gap")}
                self.inj.update(live)

    def set_detectors(self, changes: dict):
        with self.lock:
            staged = copy.deepcopy(self.det_params)
            for meth, kw in changes.items():
                if meth not in METHODS:
                    raise ValueError(f"unknown method '{meth}', use one of {list(METHODS)}")
                if not isinstance(kw, dict):
                    raise ValueError(f"detectors.{meth} must be an object of parameters, got {kw!r}")
                staged[meth].update(kw)
                staged[meth] = make(meth, **staged[meth]).params     # validates, raises ValueError
            self.det_params = staged
            if self.dets:
                for meth in changes:
                    self.dets[meth].set_params(**{k: staged[meth][k] for k in changes[meth]})

    def auto_threshold(self, methods=None, target=None, source=None):
        with self.lock:
            if self.cal is None:
                raise ValueError("calibration.json not found; run python calibrate.py")
            target = int(target or self.target)
            source = source or self.cal_source
            if str(target) not in {str(t) for t in self.cal["targets"]}:
                raise ValueError(f"target ARL0 must be one of {self.cal['targets']}")
            out, warnings = {}, []
            for meth in (methods or list(METHODS)):
                p = self.det_params[meth]
                sp = SHAPE_PARAM[meth]
                v = calibrated_threshold(self.cal, meth, p.get(sp) if sp else None, target, source,
                                         extra={"beta": p.get("beta", 0.1)})
                out[meth] = {THRESHOLD_PARAM[meth]: float(f"{v:.6g}")}
                rng = calibrated_shape_range(self.cal, meth, source)
                if rng and not rng[0] <= p[sp] <= rng[1]:
                    warnings.append(f"{meth}: {sp} = {p[sp]:g} is outside the calibrated range "
                                    f"{rng[0]:g} to {rng[1]:g}; the threshold of the nearest edge was used, "
                                    f"so the real ARL0 will differ from {target}")
            self.target, self.cal_source = target, source
            self.set_detectors(out)
            return {"set": out, "warnings": warnings}

    def reset_detectors(self):
        with self.lock:
            if self.dets:
                for d in self.dets.values():
                    d.reset()

    def trigger(self, kind, sensor, magnitude=None, duration=None, direction="up", hold=0):
        with self.lock:
            if sensor not in self.S:
                raise ValueError(f"unknown sensor '{sensor}', use one of {self.S}")
            if self.idx < self.B:
                raise ValueError(f"wait until the baseline window is over (cycle {self.B})")
            self.inj.trigger(kind, [sensor], magnitude=magnitude, duration=duration, direction=direction, hold=hold)

    def replay(self, action=None, engine=None, speed=None, loop=None):
        with self.lock:
            if speed is not None:
                if not 0.05 <= speed <= MAX_SPEED:
                    raise ValueError(f"speed must be between 0.05 and {MAX_SPEED} cycles/s "
                                     f"(Prometheus scrapes every {SCRAPE_INTERVAL_S:g} s; faster replay loses points)")
                self.speed = float(speed)
            if loop is not None:
                self.loop = bool(loop)
            if engine is not None and action is None:
                action = "select"
            if action == "select" and engine is None:
                raise ValueError("action 'select' needs an engine number")
            if action == "play":
                self.playing = True
            elif action == "pause":
                self.playing = False
            elif action == "restart":
                self.load_engine(self.unit)
            elif action == "next":
                self.load_engine(self.units[(self.units.index(self.unit) + 1) % len(self.units)])
            elif action == "select":
                self.load_engine(int(engine))
            elif action is not None:
                raise ValueError("action must be play, pause, restart, next or select")

    def preset(self, name):
        if name not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS)}")
        self.set_injector({"kinds": copy.deepcopy(PRESETS[name])})

    def status(self):
        with self.lock:
            active = [dict(sensor=e.sensor, kind=e.kind, start=e.start, end=e.end,
                           magnitude=round(e.magnitude, 2), source=e.source)
                      for e in self.inj.events if e.start <= self.idx and (e.end is None or self.idx <= e.end)]
            return dict(engine=self.unit, cycle=self.idx, life=self.life, playing=self.playing,
                        speed=self.speed, in_baseline=self.idx <= self.B, active_events=active,
                        recent_alarms=list(self.recent)[:15])
