"""Controlled anomaly injection for sensor streams.

The idea: take a real, clean signal and add anomalies whose type, frequency
and size we choose. Because we put them there, we know exactly where they
are, so we can measure how well CUSUM / EWMA / SPRT find them (detection
delay, false alarms, missed events).

Sizes are given in units of sigma0, the baseline noise level of each sensor.
"3" means "3 standard deviations of normal noise", which makes one setting
mean the same thing for a temperature in °R and a speed in rpm. It is also
the unit CUSUM thinks in: a shift of Delta = 2 sigma0 suggests kappa ~ 1 sigma0.

Five anomaly kinds:

  spike        one bad sample (classic outlier)                    -> slide 1
  level_shift  sudden jump that stays for a while                  -> CUSUM
  drift        slow linear ramp, then holds the offset             -> EWMA
  noise_burst  noise std multiplied by a factor, mean unchanged    -> mean
               detectors should miss this; a variance chart would not
  stuck        sensor freezes at its last value (flatline fault)

Two ways to use it:

  * live / streaming:  inj = AnomalyInjector(...); inj.step(clean_values)
    every new sample. inj.update(...) and inj.trigger(...) change the
    settings while it runs (e.g. from a web slider or a button).
  * batch:  inject_frame(df, sensors, inj) runs the same thing over a
    whole DataFrame and returns the injected data plus an events table.

Both use the same code path, so a batch run and a live replay with the same
seed and settings give identical results.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

SEGMENT_KINDS = ("level_shift", "drift", "noise_burst", "stuck")
ALL_KINDS = ("spike",) + SEGMENT_KINDS
DIRECTIONS = ("up", "down", "both", "wear")

# per_100   : expected number of new events per 100 samples, per sensor
#             (per group in "joint" mode). Only counted while the sensor is
#             free, so busy sensors get slightly fewer events than this.
# magnitude : [low, high] in sigma0 units, drawn uniformly. For noise_burst it
#             is the factor the noise std is multiplied by. Not used by stuck.
# duration  : [low, high] samples, drawn uniformly (integers). For
#             level_shift, null means "stays until the end of the stream".
# hold      : drift only. Samples the drift keeps its final offset after the ramp.
# direction : up | down | both (random sign) | wear (the direction the sensor
#             naturally moves as the engine wears out)
DEFAULT_KINDS: dict[str, dict] = {
    "spike": dict(enabled=True, per_100=2.0, magnitude=[4.0, 8.0], direction="both"),
    "level_shift": dict(enabled=True, per_100=0.5, magnitude=[1.0, 3.0],
                        duration=[20, 60], direction="both"),
    "drift": dict(enabled=True, per_100=0.3, magnitude=[2.0, 5.0],
                  duration=[30, 80], hold=[10, 30], direction="both"),
    "noise_burst": dict(enabled=False, per_100=0.3, magnitude=[2.0, 4.0], duration=[10, 30]),
    "stuck": dict(enabled=False, per_100=0.2, duration=[10, 30]),
}


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def _as_range(v, name, integer=False, allow_none=False):
    if v is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be null")
    if isinstance(v, (int, float)):
        v = [v, v]
    if len(v) != 2:
        raise ValueError(f"{name} must be a number or [low, high]")
    lo, hi = (int(v[0]), int(v[1])) if integer else (float(v[0]), float(v[1]))
    if lo > hi:
        raise ValueError(f"{name}: low ({lo}) > high ({hi})")
    if lo < 0:
        raise ValueError(f"{name} must be >= 0")
    return [lo, hi]


def _validate_kind(kind: str, spec: dict) -> dict:
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown anomaly kind '{kind}', use one of {ALL_KINDS}")
    allowed = set(DEFAULT_KINDS[kind])
    extra = set(spec) - allowed
    if extra:
        raise ValueError(f"{kind}: unknown setting(s) {sorted(extra)}; allowed {sorted(allowed)}")
    s = dict(spec)
    s["enabled"] = bool(s["enabled"])
    s["per_100"] = float(s["per_100"])
    if not 0 <= s["per_100"] <= 100:
        raise ValueError(f"{kind}.per_100 must be between 0 and 100")
    if "magnitude" in s:
        s["magnitude"] = _as_range(s["magnitude"], f"{kind}.magnitude")
        if kind == "noise_burst" and s["magnitude"][0] < 1:
            raise ValueError("noise_burst.magnitude is a std factor and must be >= 1")
    if "duration" in s:
        s["duration"] = _as_range(s["duration"], f"{kind}.duration", integer=True,
                                  allow_none=(kind == "level_shift"))
        if s["duration"] is not None and s["duration"][0] < 1:
            raise ValueError(f"{kind}.duration must be >= 1")
    if "hold" in s:
        s["hold"] = _as_range(s["hold"], f"{kind}.hold", integer=True)
    if "direction" in s and s["direction"] not in DIRECTIONS:
        raise ValueError(f"{kind}.direction must be one of {DIRECTIONS}")
    return s


@dataclass
class InjectorConfig:
    seed: int = 42
    baseline_cycles: int = 30   # clean warm-up: no anomalies, sigma0 estimated here
    min_gap: int = 10           # clean samples kept between two events on one sensor
    mode: str = "independent"   # "independent": each sensor gets its own events
                                # "joint": one event hits all sensors at once
    kinds: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_KINDS))
    scheduled: list = field(default_factory=list)  # fixed events, see _norm_scheduled

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "InjectorConfig":
        d = dict(d or {})
        kinds = copy.deepcopy(DEFAULT_KINDS)
        for k, spec in (d.pop("kinds", None) or {}).items():
            if k not in kinds:
                raise ValueError(f"unknown anomaly kind '{k}', use one of {ALL_KINDS}")
            kinds[k].update(spec or {})
        unknown = set(d) - {"seed", "baseline_cycles", "min_gap", "mode", "scheduled"}
        if unknown:
            raise ValueError(f"unknown config key(s): {sorted(unknown)}")
        cfg = cls(kinds=kinds, **d)
        cfg.validate()
        return cfg

    def validate(self):
        if self.mode not in ("independent", "joint"):
            raise ValueError("mode must be 'independent' or 'joint'")
        if self.baseline_cycles < 2:
            raise ValueError("baseline_cycles must be >= 2 (needed to estimate sigma0)")
        if self.min_gap < 0:
            raise ValueError("min_gap must be >= 0")
        self.kinds = {k: _validate_kind(k, v) for k, v in self.kinds.items()}
        self.scheduled = [_norm_scheduled(e, self.baseline_cycles) for e in (self.scheduled or [])]

    def to_dict(self) -> dict:
        return dict(seed=self.seed, baseline_cycles=self.baseline_cycles, min_gap=self.min_gap,
                    mode=self.mode, kinds=copy.deepcopy(self.kinds),
                    scheduled=copy.deepcopy(self.scheduled))


def _norm_scheduled(e: Mapping, baseline: int) -> dict:
    """A fixed event, e.g.
    {kind: level_shift, start: 120, duration: 50, magnitude: 2.5,
     direction: up, sensors: [s4, s11], units: [1, 2]}
    start is the sample number (= cycle in C-MAPSS). sensors / units
    omitted means all of them."""
    e = dict(e)
    kind = e.get("kind")
    if kind not in ALL_KINDS:
        raise ValueError(f"scheduled event: kind must be one of {ALL_KINDS}, got {kind!r}")
    allowed = {"kind", "start", "duration", "magnitude", "direction", "sensors", "units", "hold"}
    if set(e) - allowed:
        raise ValueError(f"scheduled event: unknown key(s) {sorted(set(e) - allowed)}")
    start = int(e["start"])
    if start <= baseline:
        raise ValueError(f"scheduled event start {start} is inside the baseline window "
                         f"(1..{baseline}); move it later or shorten baseline_cycles")
    dur = e.get("duration", 1 if kind == "spike" else 30)
    if kind == "spike":
        dur = 1
    if dur is not None:
        dur = int(dur)
        if dur < 1:
            raise ValueError("scheduled event duration must be >= 1")
    elif kind != "level_shift":
        raise ValueError("only level_shift may have duration: null")
    mag_default = {"spike": 6.0, "level_shift": 2.0, "drift": 4.0, "noise_burst": 3.0, "stuck": 0.0}
    out = dict(kind=kind, start=start, duration=dur,
               magnitude=float(e.get("magnitude", mag_default[kind])),
               direction=e.get("direction", "up"),
               sensors=list(e["sensors"]) if e.get("sensors") else None,
               units=[int(u) for u in e["units"]] if e.get("units") else None,
               hold=int(e.get("hold", 0)))
    if out["direction"] not in DIRECTIONS:
        raise ValueError(f"scheduled event direction must be one of {DIRECTIONS}")
    return out


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
@dataclass
class Event:
    event_id: int
    kind: str
    sensor: str
    start: int                 # first affected sample (t)
    end: int | None            # last affected sample, inclusive; None = until stream ends
    magnitude: float           # signed, sigma0 units (noise_burst: std factor; stuck: 0)
    sigma0: float
    ramp: int = 0              # drift: samples to reach full offset
    source: str = "random"     # "random" | "scheduled" | "trigger"
    stuck_value: float = math.nan

    def active(self, t: int) -> bool:
        return self.start <= t and (self.end is None or t <= self.end)

    @property
    def raw_magnitude(self) -> float:
        """Size in the sensor's own unit (e.g. °R)."""
        if self.kind == "stuck":
            return math.nan
        if self.kind == "noise_burst":
            return self.magnitude * self.sigma0      # new noise std
        return self.magnitude * self.sigma0          # offset added


# --------------------------------------------------------------------------
# the injector
# --------------------------------------------------------------------------
class AnomalyInjector:
    """Adds anomalies to a multi-sensor stream, one sample at a time.

    sensors  : names of the channels to corrupt
    config   : InjectorConfig (or dict)
    sigma0   : optional {sensor: baseline std}. If not given, it is estimated
               from the first `baseline_cycles` samples of each stream.
    wear_dir : optional {sensor: +1/-1}, used by direction "wear".
    """

    def __init__(self, sensors: Iterable[str], config: InjectorConfig | Mapping | None = None,
                 sigma0: Mapping[str, float] | None = None,
                 wear_dir: Mapping[str, int] | None = None, stream_id: int = 0):
        self.sensors = list(sensors)
        if not self.sensors:
            raise ValueError("need at least one sensor")
        if config is None or isinstance(config, Mapping):
            config = InjectorConfig.from_dict(config)
        self.cfg = config
        self.cfg.validate()
        self.fixed_sigma0 = dict(sigma0) if sigma0 else None
        if self.fixed_sigma0:
            missing = [s for s in self.sensors if s not in self.fixed_sigma0]
            if missing:
                raise ValueError(f"sigma0 missing for {missing}")
        self.wear_dir = {s: 1 for s in self.sensors}
        self.wear_dir.update(wear_dir or {})
        self._next_id = 1
        self.reset(stream_id)

    # ---- lifecycle --------------------------------------------------------
    def reset(self, stream_id: int = 0):
        """Start a new stream (e.g. the next engine). Event ids keep counting."""
        self.stream_id = stream_id
        self.rng = np.random.default_rng([int(self.cfg.seed), int(stream_id)])
        self.t = 0
        self.events: list[Event] = []
        self._active: dict[str, Event | None] = {s: None for s in self.sensors}
        self._free_from = {s: self.cfg.baseline_cycles + 1 for s in self.sensors}
        self._last_out = {s: math.nan for s in self.sensors}
        self._acc = {s: [] for s in self.sensors}
        self.sigma0 = dict(self.fixed_sigma0) if self.fixed_sigma0 else {}
        self._schedule = [e for e in self.cfg.scheduled
                          if e["units"] is None or int(stream_id) in e["units"]]

    # ---- live controls ----------------------------------------------------
    def update(self, changes: Mapping):
        """Change settings while running, e.g.
        inj.update({"kinds": {"spike": {"per_100": 5, "magnitude": [6, 10]}}, "min_gap": 5})
        Takes effect from the next sample. Events already running keep their size."""
        new = self.cfg.to_dict()
        changes = dict(changes)
        for k, spec in (changes.pop("kinds", None) or {}).items():
            if k not in new["kinds"]:
                raise ValueError(f"unknown anomaly kind '{k}'")
            new["kinds"][k].update(spec)
        for k in ("seed", "baseline_cycles", "scheduled"):
            if k in changes:
                raise ValueError(f"'{k}' cannot be changed while running; use reset with a new config")
        new.update(changes)
        self.cfg = InjectorConfig.from_dict(new)   # validates; raises on bad input

    def trigger(self, kind: str, sensors: Iterable[str] | None = None, magnitude: float | None = None,
                duration: int | None = None, direction: str = "up", hold: int = 0):
        """Force an event to start on the next sample (a "inject now" button)."""
        e = _norm_scheduled(dict(kind=kind, start=max(self.t + 1, self.cfg.baseline_cycles + 1),
                                 duration=duration if (duration is not None or kind == "level_shift")
                                 else (1 if kind == "spike" else 30),
                                 magnitude=magnitude if magnitude is not None else
                                 {"spike": 6, "level_shift": 2, "drift": 4,
                                  "noise_burst": 3, "stuck": 0}[kind],
                                 direction=direction, sensors=list(sensors) if sensors else None,
                                 hold=hold), self.cfg.baseline_cycles)
        e["source"] = "trigger"
        self._schedule.append(e)

    # ---- helpers ----------------------------------------------------------
    def _draw(self, r, integer=False):
        lo, hi = r
        if integer:
            return int(self.rng.integers(lo, hi + 1))
        return float(self.rng.uniform(lo, hi))

    def _sign(self, direction: str, sensor: str, shared: int | None = None) -> int:
        if direction == "up":
            return 1
        if direction == "down":
            return -1
        if direction == "wear":
            return int(self.wear_dir.get(sensor, 1))
        return shared if shared is not None else int(self.rng.choice([-1, 1]))

    def _slots(self) -> list[list[str]]:
        return [self.sensors] if self.cfg.mode == "joint" else [[s] for s in self.sensors]

    def _schedule_conflict(self, slot, t, length) -> bool:
        """True if a fixed event on these sensors starts before this random
        event (plus its gap) would be over. Fixed events always win."""
        last = math.inf if length is None else t + length - 1 + self.cfg.min_gap
        for e in self._schedule:
            if t <= e["start"] <= last and (e["sensors"] is None or set(slot) & set(e["sensors"])):
                return True
        return False

    def _new_event(self, kind, sensor, t, length, magnitude, ramp, source) -> Event:
        ev = Event(event_id=self._next_id, kind=kind, sensor=sensor, start=t,
                   end=None if length is None else t + length - 1,
                   magnitude=magnitude, sigma0=self.sigma0[sensor], ramp=ramp, source=source,
                   stuck_value=self._last_out[sensor] if kind == "stuck" else math.nan)
        self.events.append(ev)
        return ev

    def _start_segment(self, kind, slot, t, length, mag, direction, ramp, source):
        shared = int(self.rng.choice([-1, 1])) if direction == "both" else None
        eid = self._next_id
        for s in slot:
            prev = self._active[s]
            if prev is not None:                       # a fixed event interrupts a random one
                prev.end = t - 1
            signed = mag if kind in ("noise_burst", "stuck") else self._sign(direction, s, shared) * mag
            ev = self._new_event(kind, s, t, length, signed, ramp, source)
            ev.event_id = eid                          # same id for every sensor in the group
            self._active[s] = ev
        self._next_id = eid + 1

    def _start_spike(self, slot, t, mag, direction, source) -> dict[str, Event]:
        shared = int(self.rng.choice([-1, 1])) if direction == "both" else None
        eid = self._next_id
        out = {}
        for s in slot:
            ev = self._new_event("spike", s, t, 1, self._sign(direction, s, shared) * mag, 0, source)
            ev.event_id = eid
            out[s] = ev
        self._next_id = eid + 1
        return out

    # ---- main step --------------------------------------------------------
    def step(self, clean: Mapping[str, float]) -> dict:
        """Feed one clean sample {sensor: value}; get back the corrupted one.

        Returns {"t", "values", "labels", "kinds", "event_ids", "in_baseline"}.
        labels[s] is 1 when sensor s is inside an injected event at this sample."""
        self.t += 1
        t = self.t
        cfg = self.cfg
        x = {s: float(clean[s]) for s in self.sensors}
        res = dict(t=t, values=dict(x), labels={s: 0 for s in self.sensors},
                   kinds={s: "" for s in self.sensors}, event_ids={s: 0 for s in self.sensors},
                   in_baseline=t <= cfg.baseline_cycles)

        # 1) baseline window: leave clean, learn sigma0
        if t <= cfg.baseline_cycles:
            for s in self.sensors:
                self._acc[s].append(x[s])
                self._last_out[s] = x[s]
            if t == cfg.baseline_cycles and not self.fixed_sigma0:
                for s in self.sensors:
                    sd = float(np.std(self._acc[s], ddof=1))
                    self.sigma0[s] = sd if sd > 0 else 1e-9
            return res

        # 2) close finished segment events
        for s in self.sensors:
            ev = self._active[s]
            if ev is not None and ev.end is not None and t > ev.end:
                self._active[s] = None
                self._free_from[s] = ev.end + 1 + cfg.min_gap

        # 3) fixed (scheduled / triggered) events that start now
        spikes: dict[str, Event] = {}
        due = [e for e in self._schedule if e["start"] == t]
        for e in due:
            self._schedule.remove(e)
            slot = [s for s in (e["sensors"] or self.sensors) if s in self._active]
            if not slot:
                continue
            src = e.get("source", "scheduled")
            if e["kind"] == "spike":
                spikes.update(self._start_spike(slot, t, e["magnitude"], e["direction"], src))
            else:
                length = None if e["duration"] is None else e["duration"] + (e["hold"] if e["kind"] == "drift" else 0)
                self._start_segment(e["kind"], slot, t, length, e["magnitude"], e["direction"],
                                    e["duration"] or 0, src)

        # 4) random events
        seg_kinds = [k for k in SEGMENT_KINDS if cfg.kinds[k]["enabled"] and cfg.kinds[k]["per_100"] > 0]
        for slot in self._slots():
            free = all(self._active[s] is None and t >= self._free_from[s] for s in slot)
            if not free or any(s in spikes for s in slot):
                continue
            started = False
            for k in (self.rng.permutation(seg_kinds) if seg_kinds else []):
                k = str(k)
                spec = cfg.kinds[k]
                if self.rng.random() >= spec["per_100"] / 100.0:
                    continue
                dur = None if spec.get("duration") is None else self._draw(spec["duration"], integer=True)
                hold = self._draw(spec["hold"], integer=True) if k == "drift" else 0
                length = None if dur is None else dur + hold
                if self._schedule_conflict(slot, t, length):
                    continue
                mag = self._draw(spec["magnitude"]) if "magnitude" in spec else 0.0
                self._start_segment(k, slot, t, length, mag, spec.get("direction", "both"), dur or 0, "random")
                started = True
                break
            # spikes only on clean samples, so every sample has one clear label
            sp = cfg.kinds["spike"]
            if not started and sp["enabled"] and self.rng.random() < sp["per_100"] / 100.0 \
                    and not self._schedule_conflict(slot, t, 1):
                spikes.update(self._start_spike(slot, t, self._draw(sp["magnitude"]), sp["direction"], "random"))

        # 5) apply
        for s in self.sensors:
            v = x[s]
            ev = self._active[s]
            if ev is not None and ev.active(t):
                sig = ev.sigma0
                if ev.kind == "level_shift":
                    v += ev.magnitude * sig
                elif ev.kind == "drift":
                    frac = min((t - ev.start + 1) / max(ev.ramp, 1), 1.0)
                    v += ev.magnitude * sig * frac
                elif ev.kind == "noise_burst":
                    v += float(self.rng.normal(0.0, sig * math.sqrt(max(ev.magnitude ** 2 - 1, 0.0))))
                elif ev.kind == "stuck":
                    v = ev.stuck_value
                res["labels"][s], res["kinds"][s], res["event_ids"][s] = 1, ev.kind, ev.event_id
            if s in spikes:
                ev = spikes[s]
                v += ev.magnitude * ev.sigma0
                res["labels"][s], res["kinds"][s], res["event_ids"][s] = 1, "spike", ev.event_id
            res["values"][s] = v
            self._last_out[s] = v
        return res

    # ---- reporting --------------------------------------------------------
    def event_records(self, time_of_t: Mapping[int, object] | None = None, **extra) -> list[dict]:
        """Events of the current stream as rows. time_of_t maps step -> real time
        (e.g. cycle number). Events running past the end are cut at the last sample."""
        rows = []
        tmap = time_of_t or {}
        for ev in self.events:
            end = self.t if ev.end is None else min(ev.end, self.t)
            if end < ev.start:          # was scheduled but stream ended / was interrupted at once
                continue
            rows.append(dict(**extra, event_id=ev.event_id, sensor=ev.sensor, kind=ev.kind,
                             source=ev.source,
                             start=tmap.get(ev.start, ev.start), end=tmap.get(end, end),
                             length=end - ev.start + 1,
                             magnitude_sigma=ev.magnitude, magnitude_raw=ev.raw_magnitude,
                             sigma0=ev.sigma0,
                             ramp=ev.ramp if ev.kind == "drift" else None,
                             cut_by_stream_end=(ev.end is None or ev.end > self.t)))
        return rows


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------
def inject_frame(df: pd.DataFrame, sensors: list[str], injector: AnomalyInjector,
                 unit_col: str = "unit", time_col: str = "cycle") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the injector over every unit (engine) in df.

    Returns (data, events):
      data   : df with each sensor replaced by its corrupted value, plus
               <s>_clean, <s>_label (0/1), <s>_kind, <s>_event, and any_label
      events : one row per (event, sensor) with start/end/size
    """
    parts, events = [], []
    for unit, g in df.groupby(unit_col, sort=True):
        g = g.sort_values(time_col)
        injector.reset(stream_id=int(unit))
        vals = {s: [] for s in sensors}
        lab, kind, eid = ({s: [] for s in sensors} for _ in range(3))
        tmap = {}
        clean_rows = g[sensors].to_dict("records")
        for i, (row, tval) in enumerate(zip(clean_rows, g[time_col].tolist()), start=1):
            r = injector.step(row)
            tmap[i] = tval
            for s in sensors:
                vals[s].append(r["values"][s])
                lab[s].append(r["labels"][s])
                kind[s].append(r["kinds"][s])
                eid[s].append(r["event_ids"][s])
        out = g.copy()
        for s in sensors:
            out[f"{s}_clean"] = g[s].values
            out[s] = vals[s]
            out[f"{s}_label"] = lab[s]
            out[f"{s}_kind"] = kind[s]
            out[f"{s}_event"] = eid[s]
        out["any_label"] = out[[f"{s}_label" for s in sensors]].max(axis=1)
        parts.append(out)
        events.extend(injector.event_records(tmap, unit=int(unit)))
    data = pd.concat(parts, ignore_index=True)
    ev_cols = ["unit", "event_id", "sensor", "kind", "source", "start", "end", "length",
               "magnitude_sigma", "magnitude_raw", "sigma0", "ramp", "cut_by_stream_end"]
    return data, pd.DataFrame(events, columns=ev_cols)
