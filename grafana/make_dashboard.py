"""Builds grafana/dashboards/detection.json (run again after editing).

    python grafana/make_dashboard.py
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
COL = {"shewhart": "#7f7f7f", "sprt": "#9467bd", "cusum": "#1f77b4", "ewma": "#ff7f0e", "mewma": "#2ca02c"}
_id = [0]


def nid():
    _id[0] += 1
    return _id[0]


def tgt(expr, legend, ref="A", instant=False):
    t = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "range": not instant}
    if instant:
        t["instant"] = True
    return t


def dashed(regex, color):
    return {"matcher": {"id": "byRegexp", "options": regex},
            "properties": [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 6]}},
                           {"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
                           {"id": "custom.lineWidth", "value": 2}]}


def color(regex, c):
    return {"matcher": {"id": "byRegexp", "options": regex},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}]}


def ts(title, targets, x, y, w=12, h=8, unit="none", overrides=(), desc="", axis=""):
    return {"id": nid(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
            "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 1, "showPoints": "never",
                                                                  "spanNulls": False, "axisLabel": axis},
                                         "color": {"mode": "palette-classic"}},
                            "overrides": list(overrides)},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                        "tooltip": {"mode": "multi", "sort": "none"}}}


def stat(title, expr, x, y, w=4, h=4, unit="none", decimals=0, mappings=None, desc="", legend=""):
    return {"id": nid(), "type": "stat", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [tgt(expr, legend, instant=True)],
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals, "mappings": mappings or [],
                                         "color": {"mode": "fixed", "fixedColor": "#1f6feb"}}, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "value", "graphMode": "none", "textMode": "value_and_name"
                        if legend else "value", "justifyMode": "center"}}


def bars(title, expr, x, y, w=6, h=7, unit="none", decimals=0, desc=""):
    return {"id": nid(), "type": "bargauge", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [tgt(expr, "{{method}}", instant=True)],
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals, "min": 0,
                                         "color": {"mode": "fixed", "fixedColor": "#1f6feb"}},
                            "overrides": [color(f"^{m}$", c) for m, c in COL.items()]},
            "options": {"orientation": "horizontal", "displayMode": "basic", "showUnfilled": True,
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


panels = []
y = 0
# ---- header stats
panels += [
    stat("Engine", "engine_unit", 0, y),
    stat("Cycle", "engine_cycle", 4, y),
    stat("Life used", "engine_life_fraction", 8, y, unit="percentunit", decimals=0,
         desc="Cycle / total life of this engine. Real wear shows from about 30-40% of life."),
    stat("Detectors", "engine_in_baseline", 12, y, mappings=[
        {"type": "value", "options": {"0": {"text": "monitoring", "color": "green"},
                                      "1": {"text": "learning mu0", "color": "orange"}}}],
         desc="During the first baseline_cycles the detectors estimate mu0 and do not alarm."),
    stat("Injected events (per sensor)", "sum(injected_events_total)", 16, y),
    stat("Replay", "replay_speed_cycles_per_second * replay_playing", 20, y, unit="none", decimals=2,
         desc="Cycles per second (0 = paused)."),
]
y += 4
panels += [
    ts("Signal $sensor: raw (with injected anomalies) vs clean", [
        tgt('sensor_value{sensor="$sensor", series="raw"}', "raw (injected)"),
        tgt('sensor_value{sensor="$sensor", series="clean"}', "clean (original)", "B")], 0, y,
       overrides=[color("raw.*", "#d62728"), color("clean.*", "#1f77b4")],
       desc="Raw = what the detectors see. Where the two lines differ an anomaly was injected."),
    {"id": nid(), "type": "state-timeline", "title": "Injected events vs alarms ($sensor)", "datasource": DS,
     "gridPos": {"x": 12, "y": y, "w": 12, "h": 8},
     "description": "Top rows: injected events by kind. Lower rows: alarms of each method on this sensor. "
                    "Flags stay on for 2 s so the 1 s scrape cannot miss them.",
     "targets": [tgt('max by (kind) (injected_event_active{sensor="$sensor"})', "injected {{kind}}"),
                 tgt('max by (method) (detector_alarm{sensor="$sensor", method=~"$method"})', "alarm {{method}}", "B")],
     "fieldConfig": {"defaults": {
         "color": {"mode": "thresholds"},
         "thresholds": {"mode": "absolute", "steps": [{"color": "transparent", "value": None},
                                                      {"color": "#d62728", "value": 0.5}]}},
         "overrides": [{"matcher": {"id": "byRegexp", "options": "injected.*"},
                        "properties": [{"id": "thresholds", "value": {"mode": "absolute", "steps": [
                            {"color": "transparent", "value": None}, {"color": "#ff9830", "value": 0.5}]}}]}]},
     "options": {"showValue": "never", "rowHeight": 0.8, "mergeValues": True, "alignValue": "left",
                 "legend": {"showLegend": False}}},
]
y += 8
panels += [
    ts("CUSUM on $sensor: C+ and C- vs h", [
        tgt('detector_statistic{method="cusum", sensor="$sensor"}', "C{{side}}"),
        tgt('detector_threshold{method="cusum", sensor="$sensor", side="h"}', "h", "B")], 0, y,
       overrides=[dashed("^h$", "#d62728"), color("Cup", "#1f77b4"), color("Cdown", "#17becf")],
       axis="sigma0 units", desc="Alarm when C+ or C- crosses h; both restart at 0 after an alarm."),
    ts("EWMA on $sensor: Z with finite-time limits", [
        tgt('detector_statistic{method="ewma", sensor="$sensor"}', "Z"),
        tgt('detector_threshold{method="ewma", sensor="$sensor"}', "limit {{side}}", "B")], 12, y,
       overrides=[dashed("limit.*", "#d62728"), color("^Z$", "#ff7f0e")], axis="sigma0 units",
       desc="Limits widen from L*lambda at the start to L*sqrt(lambda/(2-lambda)); they restart after an alarm."),
]
y += 8
panels += [
    ts("SPRT on $sensor: S_t vs log A and log B", [
        tgt('detector_statistic{method="sprt", sensor="$sensor"}', "S {{side}}"),
        tgt('detector_threshold{method="sprt", sensor="$sensor"}', "log {{side}}", "B")], 0, y,
       overrides=[dashed("log upper", "#d62728"), dashed("log lower", "#2ca02c")], axis="log likelihood ratio",
       desc="log A (upper): decide H1 = alarm. log B (lower): decide H0, restart quietly."),
    ts("MEWMA (all 14 sensors): T2 vs h", [
        tgt('detector_statistic{method="mewma"}', "T2"),
        tgt('detector_threshold{method="mewma"}', "h", "B")], 12, y,
       overrides=[dashed("^h$", "#d62728"), color("T2", "#2ca02c")], axis="T2",
       desc="One statistic for all sensors. The alarm is shown on the sensor with the largest smoothed deviation."),
]
y += 8
panels += [
    ts("Shewhart on $sensor: z vs +-k", [
        tgt('detector_statistic{method="shewhart", sensor="$sensor"}', "z"),
        tgt('detector_threshold{method="shewhart", sensor="$sensor"}', "k {{side}}", "B")], 0, y,
       overrides=[dashed("k.*", "#d62728"), color("^z$", "#7f7f7f")], axis="sigma0 units",
       desc="No memory: alarms on single extreme values (spikes)."),
    ts("Selected method(s) on $sensor: statistic vs threshold", [
        tgt('detector_statistic{method=~"$method", sensor=~"$sensor|all"}', "{{method}} {{side}}"),
        tgt('detector_threshold{method=~"$method", sensor=~"$sensor|all"}', "{{method}} limit {{side}}", "B")],
       12, y, overrides=[dashed(".*limit.*", "#d62728")],
       desc="Use the 'method' variable at the top to choose. Units differ between methods."),
]
y += 8
panels += [
    bars("True alarms (inside an injected event + grace)",
         'sum by (method) (detector_alarms_total{outcome="true", method=~"$method"})', 0, y),
    bars("False alarms (healthy part of life, no event)",
         'sum by (method) (detector_alarms_total{outcome="false", method=~"$method"})', 6, y),
    bars("Wear-zone alarms (late life, no event)",
         'sum by (method) (detector_alarms_total{outcome="wear", method=~"$method"})', 12, y,
         desc="Alarms after clean_life_fraction of life that are not explained by an injected event: real degradation."),
    bars("Mean detection delay (cycles, spikes excluded)",
         'sum by (method) (detection_delay_cycles_sum{method=~"$method", kind!="spike"}) / '
         'sum by (method) (detection_delay_cycles_count{method=~"$method", kind!="spike"})', 18, y, decimals=1,
         desc="Spikes are left out: an alarm on a spike always has delay 0 and would hide the delay on real changes."),
]
y += 7
panels += [
    bars("Detection rate (events detected / injected, per sensor)",
         'sum by (method) (detected_events_total{method=~"$method", kind!="spike"}) / ignoring(method) group_left '
         'sum(injected_events_total{kind!="spike"})', 0, y, w=12, unit="percentunit", decimals=0,
         desc="Spikes excluded: they are anomalies, not changes."),
    stat("Engines replayed", "engines_replayed_total", 12, y, w=4, h=7),
    stat("Injected events by kind", "sum by (kind) (injected_events_total)", 16, y, w=8, h=7, legend="{{kind}}"),
]

dash = {
    "uid": "cmapss-detect", "title": "Sequential change detection", "tags": ["cmapss", "cusum", "ewma"],
    "timezone": "browser", "schemaVersion": 39, "version": 1, "editable": False, "graphTooltip": 1,
    "refresh": "1s", "time": {"from": "now-5m", "to": "now"},
    "timepicker": {"refresh_intervals": ["1s", "2s", "5s", "10s", "30s"]},
    "templating": {"list": [
        {"name": "sensor", "label": "sensor", "type": "query", "datasource": DS,
         "query": {"query": "label_values(sensor_value, sensor)", "refId": "sensor"},
         "definition": "label_values(sensor_value, sensor)", "refresh": 1, "sort": 3,
         "current": {"text": "s4", "value": "s4"}},
        {"name": "method", "label": "method", "type": "custom", "multi": True, "includeAll": True,
         "query": "shewhart,sprt,cusum,ewma,mewma", "allValue": "shewhart|sprt|cusum|ewma|mewma",
         "current": {"text": ["All"], "value": ["$__all"]},
         "options": [{"text": "All", "value": "$__all", "selected": True}] +
                    [{"text": m, "value": m, "selected": False} for m in COL]},
    ]},
    "annotations": {"list": []},
    "panels": panels,
}
out = Path(__file__).parent / "dashboards" / "detection.json"
out.write_text(json.dumps(dash, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(panels)} panels)")
