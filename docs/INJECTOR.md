# The anomaly injector (stage 2)

Adds controlled anomalies to the NASA C-MAPSS turbofan engine data, so the
sequential detectors (SPRT, CUSUM, EWMA) can be tested against a known truth.

## Why inject instead of using the data as it is

The real data has no labelled outliers. If a detector raises an alarm, you
cannot tell whether it is right. When we add the anomalies ourselves, we know
the exact cycle, sensor, type and size of each one, so we can measure:

- **detection delay**: first alarm cycle minus event start
- **false alarms**: alarms where no event is active (and ARL0, slide 1)
- **missed events**: events that end with no alarm

## Files

| file | what it is |
|---|---|
| `injector.py` | the generator. Works sample by sample (for the live app) and in batch |
| `cmapss.py` | loads C-MAPSS files, sensor names, baseline noise (sigma0), wear direction |
| `inject.py` | command line: data + config in, injected CSV + events CSV out |
| `config.yaml` | all the knobs, with comments |
| `config_demo.yaml` | one fixed example of each anomaly type on engine 1 |
| `plot_injection.py` | plots one engine: clean vs injected, events shaded |
| `test_injector.py` | 12 checks that the output matches the settings |
| `data/train_FD001.txt` | FD001: 100 engines, one operating condition, run to failure |

## Quick start

From the project folder (after `python run.py setup`, see the main README):

```bash
python run.py injector-demo
```

This runs the two commands below and writes `out_demo/unit1.png`. The same by hand:

```bash
python inject.py --data data/train_FD001.txt --config config_demo.yaml --units 1 --sensors s4 s11 s7 --life-fraction 0.85 --out out_demo
python plot_injection.py --dir out_demo --unit 1 --sensors s4 s11 s7
```

Full random run on all 100 engines and the 14 sensors that move:

```bash
python inject.py --data data/train_FD001.txt --config config.yaml --out out
```

(If you did not use `run.py setup`, run these with the Python of `.venv`, for example
`.venv\Scripts\python inject.py ...` on Windows or `.venv/bin/python inject.py ...` on macOS and Linux.)

## The anomaly types

| kind | what happens | which method should catch it |
|---|---|---|
| `spike` | one sample jumps by m·sigma0 | outlier, not a change (slide 1). A good change detector should mostly *ignore* it |
| `level_shift` | +m·sigma0 for `duration` cycles | CUSUM (slide 2) |
| `drift` | ramps to m·sigma0 over `duration`, holds for `hold` | EWMA (slide 3) |
| `noise_burst` | noise std becomes m·sigma0, mean unchanged | mean detectors miss it. Good for showing a limit |
| `stuck` | sensor freezes at its last value | variance drops to 0 |

## The controls

- **How often**: `per_100` = expected new events per 100 cycles per sensor.
  A sensor that is already inside an event cannot start a new one, so the real
  count is a bit lower than the setting when many types are on.
- **How big**: `magnitude: [low, high]` in sigma0 units, drawn uniformly.
  sigma0 is the normal noise of that sensor, taken from the first
  `baseline_cycles` of every engine. So `2` means the same thing for a
  temperature (sigma0 about 4 °R for s4) and a pressure (about 0.1 psia for s11).
- **How long**: `duration: [low, high]` cycles. `null` on `level_shift` = stays until the end.
- **Which way**: `direction: up | down | both | wear`. `wear` pushes each sensor
  the way it naturally moves when the engine degrades (learned from the data),
  so a joint fault looks physically plausible.
- **Where**: `--sensors`, `--units`, and `mode: joint` to hit several sensors at the same cycle
  (needed for multivariate EWMA).
- **Exactly where**: `scheduled:` list for fixed events at given cycles (repeatable demos).
- **Same result again**: `seed`.

## Output

`injected.csv`: original columns, where each chosen sensor now holds the
corrupted value, plus for every sensor `s`:

- `s_clean` original value
- `s_label` 1 if inside an injected event
- `s_kind` event type, `s_event` event id
- `any_label` 1 if any sensor is inside an event

`events.csv`: one row per event and sensor: `unit, event_id, sensor, kind,
source, start, end, length, magnitude_sigma, magnitude_raw, sigma0, ramp,
cut_by_stream_end`.

## Using it live (for the Prometheus / Grafana app)

```python
from injector import AnomalyInjector
inj = AnomalyInjector(sensors, config, sigma0=sigma0, wear_dir=wear)
inj.reset(stream_id=engine_number)
for row in engine_rows:                       # one cycle at a time
    r = inj.step(row)                         # r["values"], r["labels"], r["kinds"]
    ...                                       # feed r["values"] to CUSUM / EWMA / SPRT

inj.update({"kinds": {"spike": {"per_100": 5, "magnitude": [6, 10]}}})   # slider moved
inj.trigger("level_shift", sensors=["s4"], magnitude=2, duration=40)      # "inject now" button
```

Batch and live use the same code, so with the same seed and settings they give
identical values (tested).

## Things to know

- **Real wear is in the data.** In the training set every engine runs until
  it fails, and the last part of each series shows real degradation (s4 climbs
  at the end of engine 1). Detectors will alarm there too, and that is a *real*
  change, not a false alarm. Either cut it with `--life-fraction 0.6`, or report
  it as a separate result ("detected natural wear at cycle X"). Stage 3 found
  that the wear is already visible from about 30% of life
  (see [STAGE3_parameters_and_ui.md](STAGE3_parameters_and_ui.md), section 5).
- **Short series.** Engines in FD001 last 128 to 362 cycles. After the 30 cycle
  baseline that is room for only a handful of segment events per sensor.
- **Fixed sigma0.** By default (`--sigma fleet`) sigma0 is pooled over all 100
  engines, which is more stable than 30 samples of one engine. With
  `--sigma stream` each engine estimates its own, like a detector would live.
- 7 of the 21 sensors are constant in FD001 (s1, s5, s6, s10, s16, s18, s19) and
  are skipped automatically.

Data: A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation
Data Set", NASA Ames Prognostics Data Repository.
