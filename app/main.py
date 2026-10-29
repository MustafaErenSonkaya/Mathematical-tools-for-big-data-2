"""FastAPI app: control page, REST API and Prometheus /metrics.

    uvicorn app.main:app --port 8000          (from the project folder)

Endpoints
    GET  /                 control page (static/index.html)
    GET  /metrics          Prometheus metrics
    GET  /api/config       all current settings
    POST /api/config       change settings: {"injector": {...}, "detectors": {...}, "scoring": {"grace": n}}
    POST /api/trigger      start an event now: {"kind", "sensor", "magnitude", "duration", "direction"}
    POST /api/replay       {"action": play|pause|restart|next|select, "engine", "speed", "loop"}
    POST /api/auto_threshold  {"methods": [...], "target_arl0": 500, "source": "iid_est"}
    POST /api/reset_detectors
    POST /api/preset       {"name": calm|cusum_demo|ewma_demo|spikes_only}
    GET  /api/status       cycle, active events, recent alarms (for the page)
Invalid input returns HTTP 400 with {"error": "readable message"}.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from .live import LiveSystem

STATIC = Path(__file__).parent / "static"
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000/d/cmapss-detect/sequential-change-detection")

system: LiveSystem | None = None


@asynccontextmanager
async def lifespan(app):
    global system
    system = LiveSystem()
    if os.environ.get("NO_REPLAY") != "1":
        system.start()
    yield
    system.stop()


app = FastAPI(title="C-MAPSS sequential change detection", lifespan=lifespan)


# ---------------------------------------------------------------- errors
@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e["loc"] if x != "body")
        msgs.append(f"{loc}: {e['msg']}" if loc else e["msg"])
    return JSONResponse(status_code=400, content={"error": "; ".join(msgs)})


# ---------------------------------------------------------------- request models
class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConfigIn(Strict):
    injector: Optional[dict] = None
    detectors: Optional[dict[str, dict]] = None
    scoring: Optional[dict] = None


class TriggerIn(Strict):
    kind: Literal["spike", "level_shift", "drift", "noise_burst", "stuck"]
    sensor: str
    magnitude: Optional[float] = Field(None, ge=0, le=20)
    duration: Optional[int] = Field(None, ge=1, le=300)
    direction: Literal["up", "down", "both", "wear"] = "up"
    hold: int = Field(0, ge=0, le=300)


class ReplayIn(Strict):
    action: Optional[Literal["play", "pause", "restart", "next", "select"]] = None
    engine: Optional[int] = None
    speed: Optional[float] = None
    loop: Optional[bool] = None


class AutoIn(Strict):
    methods: Optional[list[Literal["shewhart", "sprt", "cusum", "ewma", "mewma"]]] = None
    target_arl0: Optional[int] = None
    source: Optional[Literal["iid", "iid_est", "real_early", "real"]] = None


class PresetIn(Strict):
    name: str


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(system.m.registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/config")
def get_config():
    return dict(system.config(), grafana_url=GRAFANA_URL)


@app.post("/api/config")
def post_config(body: ConfigIn):
    if body.injector:
        system.set_injector(body.injector)
    if body.detectors:
        system.set_detectors(body.detectors)
    if body.scoring:
        bad = set(body.scoring) - {"grace"}
        if bad:
            raise ValueError(f"scoring: only 'grace' can be changed, not {sorted(bad)}")
        g = int(body.scoring["grace"])
        if not 0 <= g <= 100:
            raise ValueError("scoring.grace must be between 0 and 100 cycles")
        system.grace = g
    return get_config()


@app.post("/api/trigger")
def trigger(body: TriggerIn):
    system.trigger(body.kind, body.sensor, body.magnitude, body.duration, body.direction, body.hold)
    return {"ok": True, "message": f"{body.kind} on {body.sensor} starts at the next cycle"}


@app.post("/api/replay")
def replay(body: ReplayIn):
    system.replay(body.action, body.engine, body.speed, body.loop)
    return system.config()["replay"]


@app.post("/api/auto_threshold")
def auto_threshold(body: AutoIn):
    return system.auto_threshold(body.methods, body.target_arl0, body.source)


@app.post("/api/reset_detectors")
def reset_detectors():
    system.reset_detectors()
    return {"ok": True}


@app.post("/api/preset")
def preset(body: PresetIn):
    system.preset(body.name)
    return get_config()


@app.get("/api/status")
def status():
    return system.status()
