"""API tests for the live app (replay thread off, cycles stepped by hand).
Run:  python -m pytest -q test_app.py"""
import os

os.environ["NO_REPLAY"] = "1"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def run(n):
    for _ in range(n):
        main.system.step()


def test_config_and_metrics(client):
    c = client.get("/api/config").json()
    assert len(c["sensors"]) == 14 and c["replay"]["max_speed"] == 1.0
    run(60)
    text = client.get("/metrics").text
    for name in ("engine_cycle", "engine_unit", "sensor_value{", "detector_statistic{", "detector_threshold{",
                 "detector_alarm{", "injected_event_active{", "detector_alarms_total", "detection_delay_cycles"):
        assert name in text, name
    assert 'series="raw"' in text and 'method="mewma"' in text


def test_invalid_values_give_readable_errors(client):
    r = client.post("/api/config", json={"detectors": {"cusum": {"h": -1}}})
    assert r.status_code == 400 and "cusum.h" in r.json()["error"]
    r = client.post("/api/replay", json={"speed": 5})
    assert r.status_code == 400 and "scrape" in r.json()["error"]
    r = client.post("/api/config", json={"injector": {"kinds": {"spike": {"per_100": 500}}}})
    assert r.status_code == 400 and "per_100" in r.json()["error"]
    r = client.post("/api/trigger", json={"kind": "level_shift", "sensor": "nope"})
    assert r.status_code == 400 and "unknown sensor" in r.json()["error"]
    r = client.post("/api/trigger", json={"kind": "explode", "sensor": "s4"})
    assert r.status_code == 400 and "kind" in r.json()["error"]


def test_trigger_shift_is_detected(client):
    client.post("/api/preset", json={"name": "calm"})
    client.post("/api/replay", json={"action": "select", "engine": 3})
    run(40)
    assert client.post("/api/trigger", json={"kind": "level_shift", "sensor": "s4", "magnitude": 4,
                                            "duration": 30}).status_code == 200
    run(30)
    alarms = [a for a in client.get("/api/status").json()["recent_alarms"] if a["sensor"] == "s4"]
    assert any(a["outcome"] == "true" and a["method"] == "cusum" for a in alarms)


def test_auto_threshold_and_sliders(client):
    client.post("/api/config", json={"detectors": {"cusum": {"kappa": 1.0}}})
    j = client.post("/api/auto_threshold", json={"methods": ["cusum"], "target_arl0": 500}).json()
    assert 4 < j["set"]["cusum"]["h"] < 12            # kappa 1 needs a lower h than kappa 0.5
    assert client.get("/api/config").json()["detectors"]["cusum"]["h"] == j["set"]["cusum"]["h"]


def test_seed_restart_and_presets(client):
    r = client.post("/api/config", json={"injector": {"seed": 5}})
    assert r.status_code == 200 and r.json()["injector"]["seed"] == 5 and r.json()["replay"]["cycle"] == 0
    for p in ("calm", "cusum_demo", "ewma_demo", "spikes_only"):
        assert client.post("/api/preset", json={"name": p}).status_code == 200
    assert client.get("/").status_code == 200


# ---- regression tests for bugs found in review ------------------------------
def test_bad_input_is_400_not_500(client):
    assert client.post("/api/replay", json={"action": "select"}).status_code == 400
    r = client.post("/api/config", json={"injector": {"kinds": {"spike": 5}}})
    assert r.status_code == 400 and "injector.kinds.spike" in r.json()["error"]
    r = client.post("/api/config", json={"injector": {"kinds": [1, 2]}})
    assert r.status_code == 400 and "injector.kinds" in r.json()["error"]


def test_params_are_stored_as_numbers(client):
    client.post("/api/config", json={"detectors": {"cusum": {"h": "12"}}})
    assert client.get("/api/config").json()["detectors"]["cusum"]["h"] == 12.0


def test_auto_threshold_warns_outside_calibrated_grid(client):
    client.post("/api/config", json={"detectors": {"ewma": {"lam": 0.01}}})
    j = client.post("/api/auto_threshold", json={"methods": ["ewma"]}).json()
    assert j["warnings"] and "outside the calibrated range" in j["warnings"][0]
    client.post("/api/config", json={"detectors": {"ewma": {"lam": 0.1}}})
    assert client.post("/api/auto_threshold", json={"methods": ["ewma"]}).json()["warnings"] == []


def test_clip_and_restart_controls(client):
    d = client.post("/api/config", json={"detectors": {"cusum": {"clip": 4, "restart": "continue"}}}).json()
    assert d["detectors"]["cusum"]["clip"] == 4.0 and d["detectors"]["cusum"]["restart"] == "continue"
    d = client.post("/api/config", json={"detectors": {"cusum": {"clip": None, "restart": "reset"}}}).json()
    assert d["detectors"]["cusum"]["clip"] is None


def _metric(client, name):
    line = next(x for x in client.get("/metrics").text.splitlines() if x.startswith(name + " "))
    return float(line.split()[1])


def test_engine_end_counted_once(client):
    s = main.system
    client.post("/api/replay", json={"action": "select", "engine": 1, "loop": False})
    before = _metric(client, "engines_replayed_total")
    for _ in range(s.life + 5):                     # a few extra steps = Play pressed again at the end
        s.step()
    assert s.playing is False
    assert _metric(client, "engines_replayed_total") == before + 1


def test_flags_clear_while_paused(client):
    import time
    s = main.system
    client.post("/api/preset", json={"name": "calm"})
    client.post("/api/replay", json={"action": "select", "engine": 5})
    client.post("/api/replay", json={"action": "pause"})
    for _ in range(40):
        s.step()
    s.trigger("spike", "s4", 9)
    s.step()
    assert s._held                                  # the spike and Shewhart's alarm are flagged
    th = s.start()                                  # replay thread, but paused
    time.sleep(3.2)
    assert not s._held                              # flags switched off after 2 s
    s.stop()
    th.join(timeout=3)
