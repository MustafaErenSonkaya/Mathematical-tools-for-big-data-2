"""Screenshots of the control page and the Grafana dashboard for the docs.

Needs the stack running (docker compose up) and Playwright:
    pip install playwright && python -m playwright install chromium
    python docs/take_screenshots.py --app http://localhost:8000 --grafana http://localhost:3000

It selects engine 1 with the CUSUM demo preset, injects a 2 sigma0 level
shift on s4 at cycle 40, waits until about cycle 100 and takes the pictures.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "img"


def api(base, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def wait_cycle(base, c):
    while api(base, "/api/status")["cycle"] < c:
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="http://localhost:8000")
    ap.add_argument("--grafana", default="http://localhost:3000")
    ap.add_argument("--until", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    api(a.app, "/api/preset", {"name": "cusum_demo"})
    api(a.app, "/api/replay", {"action": "select", "engine": 1, "speed": 1.0})
    wait_cycle(a.app, 40)
    api(a.app, "/api/trigger", {"kind": "level_shift", "sensor": "s4", "magnitude": 2, "duration": 40,
                                "direction": "up"})
    wait_cycle(a.app, a.until)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 1000})
        pg.goto(a.app)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "control_page.png"), full_page=True)
        pg = b.new_page(viewport={"width": 1700, "height": 2400})
        pg.goto(f"{a.grafana}/d/cmapss-detect/sequential-change-detection?var-sensor=s4&from=now-3m&to=now&kiosk")
        pg.wait_for_timeout(8000)
        pg.screenshot(path=str(OUT / "grafana_dashboard.png"), full_page=True)
        b.close()
    print("wrote", list(OUT.glob("*.png")))


if __name__ == "__main__":
    main()
