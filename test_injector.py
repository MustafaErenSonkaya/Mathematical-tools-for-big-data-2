"""Checks that the injector does what the settings say.
Run:  python -m pytest -q test_injector.py   (or: python test_injector.py)"""
import numpy as np
import pandas as pd
import pytest

from injector import AnomalyInjector, InjectorConfig, inject_frame

OFF = {k: {"enabled": False} for k in ("spike", "level_shift", "drift", "noise_burst", "stuck")}


def cfg(scheduled=None, **kinds):
    k = {name: dict(v) for name, v in OFF.items()}
    for name, spec in kinds.items():
        k[name].update(spec)
    return {"seed": 1, "baseline_cycles": 30, "min_gap": 5, "kinds": k, "scheduled": scheduled or []}


def stream(n, sensors=("a",), seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"unit": 1, "cycle": np.arange(1, n + 1),
                         **{s: 100 + rng.normal(0, sd, n) for s in sensors}})


def run(df, conf, sensors=("a",), sigma0=None):
    inj = AnomalyInjector(list(sensors), conf, sigma0=sigma0 or {s: 1.0 for s in sensors})
    return inject_frame(df, list(sensors), inj)


def test_baseline_untouched():
    d, _ = run(stream(500), cfg(spike={"enabled": True, "per_100": 50}))
    head = d[d.cycle <= 30]
    assert (head.a == head.a_clean).all() and head.a_label.sum() == 0


def test_spike_rate_and_size():
    d, ev = run(stream(100_030), cfg(spike={"enabled": True, "per_100": 3, "magnitude": [4, 6]}))
    rate = d[d.cycle > 30].a_label.mean() * 100
    assert abs(rate - 3) < 0.25, rate
    diff = (d.a - d.a_clean)[d.a_label == 1].abs()
    assert diff.min() >= 4 - 1e-9 and diff.max() <= 6 + 1e-9


def test_level_shift_exact():
    conf = cfg(level_shift={"enabled": True, "per_100": 2, "magnitude": [2.5, 2.5],
                            "duration": [40, 40], "direction": "up"})
    d, ev = run(stream(3000), conf)
    diff = d.a - d.a_clean
    assert np.allclose(diff[d.a_label == 1], 2.5)
    assert np.allclose(diff[d.a_label == 0], 0)
    assert (ev.length[~ev.cut_by_stream_end] == 40).all()


def test_drift_ramp_then_hold():
    conf = cfg(scheduled=[{"kind": "drift", "start": 100, "duration": 20, "hold": 10,
                           "magnitude": 4, "direction": "down"}])
    d, ev = run(stream(300), conf)
    diff = (d.a - d.a_clean).values
    seg = diff[99:129]                               # cycles 100..129
    assert np.allclose(seg[:20], -4 * np.arange(1, 21) / 20)
    assert np.allclose(seg[20:], -4)
    assert np.allclose(diff[129:], 0) and np.allclose(diff[:99], 0)


def test_noise_burst_std():
    conf = cfg(scheduled=[{"kind": "noise_burst", "start": 100, "duration": 20000, "magnitude": 3}])
    d, _ = run(stream(20200), conf)
    inside = d[d.a_label == 1].a
    assert abs(inside.std() - 3.0) < 0.1           # total std = factor * sigma0
    assert abs(inside.mean() - 100) < 0.1          # mean unchanged


def test_stuck_is_flat():
    conf = cfg(scheduled=[{"kind": "stuck", "start": 50, "duration": 25}])
    d, _ = run(stream(200), conf)
    seg = d[d.a_label == 1].a
    assert len(seg) == 25 and seg.nunique() == 1
    assert seg.iloc[0] == d.a.iloc[48]             # value of cycle 49


def test_permanent_shift_until_end():
    conf = cfg(scheduled=[{"kind": "level_shift", "start": 60, "duration": None, "magnitude": 2}])
    d, ev = run(stream(150), conf)
    assert d[d.cycle >= 60].a_label.all() and ev.cut_by_stream_end.all() and ev.end.iloc[0] == 150


def test_joint_mode_hits_all_sensors_together():
    conf = cfg(level_shift={"enabled": True, "per_100": 2, "duration": [10, 20]})
    conf["mode"] = "joint"
    s = ("a", "b", "c")
    d, ev = run(stream(3000, s), conf, s)
    assert (d.a_label == d.b_label).all() and (d.b_label == d.c_label).all()
    assert (ev.groupby("event_id").sensor.nunique() == 3).all()


def test_wear_direction():
    conf = cfg(scheduled=[{"kind": "level_shift", "start": 40, "duration": 10, "magnitude": 2,
                           "direction": "wear"}])
    inj = AnomalyInjector(["a", "b"], conf, sigma0={"a": 1, "b": 1}, wear_dir={"a": 1, "b": -1})
    d, _ = inject_frame(stream(80, ("a", "b")), ["a", "b"], inj)
    w = d[d.a_label == 1]
    assert np.allclose(w.a - w.a_clean, 2) and np.allclose(w.b - w.b_clean, -2)


def test_same_seed_same_result_and_batch_equals_live():
    conf = cfg(spike={"enabled": True}, level_shift={"enabled": True}, drift={"enabled": True},
               noise_burst={"enabled": True}, stuck={"enabled": True})
    df = stream(2000)
    d1, _ = run(df, conf)
    d2, _ = run(df, conf)
    assert d1.a.equals(d2.a)
    inj = AnomalyInjector(["a"], conf, sigma0={"a": 1.0})
    inj.reset(stream_id=1)
    live = [inj.step({"a": v})["values"]["a"] for v in df.a]
    assert np.allclose(live, d1.a)


def test_live_update_and_trigger():
    inj = AnomalyInjector(["a"], cfg(), sigma0={"a": 1.0})
    for _ in range(40):
        inj.step({"a": 0.0})
    inj.update({"kinds": {"spike": {"enabled": True, "per_100": 100, "magnitude": 5, "direction": "up"}}})
    assert inj.step({"a": 0.0})["values"]["a"] == 5.0
    inj.update({"kinds": {"spike": {"enabled": False}}})
    inj.trigger("level_shift", magnitude=3, duration=4, direction="down")
    out = [inj.step({"a": 0.0})["values"]["a"] for _ in range(6)]
    assert out == [-3.0] * 4 + [0.0, 0.0]


def test_bad_settings_rejected():
    with pytest.raises(ValueError):
        InjectorConfig.from_dict({"kinds": {"spike": {"per_100": -1}}})
    with pytest.raises(ValueError):
        InjectorConfig.from_dict({"kinds": {"spike": {"magnitude": [5, 2]}}})
    with pytest.raises(ValueError):
        InjectorConfig.from_dict({"kinds": {"spikes": {}}})
    with pytest.raises(ValueError):
        InjectorConfig.from_dict({"scheduled": [{"kind": "spike", "start": 10}]})   # inside baseline


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
