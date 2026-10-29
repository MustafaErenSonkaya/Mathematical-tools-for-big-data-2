"""One command for everything. Works the same on Windows, macOS and Linux.

    python run.py setup            install everything into .venv (once, a few minutes)
    python run.py test             run all tests (about 10 s)
    python run.py app              live app without Docker   -> http://localhost:8000
    python run.py docker           app + Prometheus + Grafana -> http://localhost:8000 and :3000
    python run.py docker-stop      stop the Docker services
    python run.py evaluate         experiments E1 to E7 (about 6 min) -> reports/
    python run.py calibrate        recompute the thresholds (about 6 min) -> calibration.json
    python run.py injector-demo    one engine with one anomaly of each kind -> out_demo/unit1.png

Extra words are passed on, for example:
    python run.py evaluate E1 E3          only experiments E1 and E3
    python run.py calibrate --quick       only the default settings (about 30 s, writes
                                          calibration_quick.json; calibration.json is kept)
    python run.py test -k cusum           only the tests with "cusum" in the name
    python run.py app --port 8001         another port
    python run.py docker --app-port 8001 --grafana-port 3001 --prometheus-port 9091
    python run.py docker --fresh          start with an empty Prometheus history
    python run.py setup --lock            the exact package versions used for the results

You never need to "activate" .venv: after setup, run.py uses it by itself.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
NEEDED = ("numpy", "pandas", "yaml", "matplotlib", "fastapi", "uvicorn", "prometheus_client", "pytest")
PY_CMD = "py" if os.name == "nt" and not shutil.which("python") else "python"


def say(msg=""):
    print(msg, flush=True)


def run(args, env=None):
    """Run a command in the project folder and return its exit code (Ctrl+C stops it cleanly)."""
    try:
        return subprocess.call([str(a) for a in args], cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 130


def take(args, flag, default=None):
    """Remove "--flag value" from args and return the value."""
    if flag in args:
        i = args.index(flag)
        if i + 1 >= len(args):
            raise SystemExit(f"{flag} needs a value")
        value = args[i + 1]
        del args[i:i + 2]
        return value
    return default


# ---------------------------------------------------------------- setup
def setup(args):
    if sys.version_info < (3, 10):
        say(f"Python 3.10 or newer is needed, this is {sys.version.split()[0]}.")
        say("Install a newer Python from https://www.python.org/downloads/ and run this again.")
        return 1
    lock = "--lock" in args
    if lock and sys.version_info < (3, 11):
        say("--lock needs Python 3.11 or newer (the exact versions used for the results).")
        return 1
    if not VENV_PY.exists():
        say(f"Creating a private Python environment in {VENV.name} ...")
        code = run([sys.executable, "-m", "venv", VENV])
        if code:
            return code
    req = "requirements-lock.txt" if lock else "requirements.txt"
    say(f"Installing the packages from {req} (a few minutes the first time) ...")
    run([VENV_PY, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    code = run([VENV_PY, "-m", "pip", "install", "--quiet", "-r", req])
    if code:
        say("\nInstalling failed. Check the internet connection and the message above.")
        return code
    say(f"\nDone. Check that everything works with:  {PY_CMD} run.py test")
    return 0


# ---------------------------------------------------------------- docker
def docker(args):
    if not shutil.which("docker"):
        say("Docker is not installed. Get Docker Desktop from https://www.docker.com/products/docker-desktop/")
        return 1
    if subprocess.run(["docker", "info"], cwd=ROOT, capture_output=True).returncode != 0:
        say("\nDocker is installed but not running. Start Docker Desktop, wait until it says it is running,")
        say("then run this command again.")
        return 1
    env = dict(os.environ)
    ports = {"APP_PORT": take(args, "--app-port", "8000"), "GRAFANA_PORT": take(args, "--grafana-port", "3000"),
             "PROM_PORT": take(args, "--prometheus-port", "9090")}
    env.update(ports)
    if "--fresh" in args:
        args.remove("--fresh")
        run(["docker", "compose", "down", "-v"], env=env)
    say("Starting... When the log says 'Application startup complete', open:")
    say(f"  control page  http://localhost:{ports['APP_PORT']}")
    say(f"  Grafana       http://localhost:{ports['GRAFANA_PORT']}")
    say(f"  Prometheus    http://localhost:{ports['PROM_PORT']}")
    say("Stop with Ctrl+C (or: python run.py docker-stop).\n")
    code = run(["docker", "compose", "up", "--build", *args], env=env)
    return 0 if code == 130 else code


def docker_stop(args):
    return run(["docker", "compose", "down", *args])


# ---------------------------------------------------------------- python commands
def app(args):
    port = take(args, "--port", "8000")
    say(f"Live app on http://localhost:{port}  (stop with Ctrl+C)")
    say("Without Docker there is no Grafana: the control page works, the dashboard link does not.\n")
    return run([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", port,
                "--no-access-log", *args])


def injector_demo(args):
    code = run([sys.executable, "inject.py", "--data", "data/train_FD001.txt", "--config", "config_demo.yaml",
                "--units", "1", "--sensors", "s4", "s11", "s7", "--life-fraction", "0.85", "--out", "out_demo"])
    if code:
        return code
    code = run([sys.executable, "plot_injection.py", "--dir", "out_demo", "--unit", "1",
                "--sensors", "s4", "s11", "s7", *args])
    if not code:
        say(f"\nOpen {Path('out_demo') / 'unit1.png'} to see the result.")
    return code


PYTHON_COMMANDS = {
    "test": lambda a: run([sys.executable, "-m", "pytest", "-q", *a]),
    "evaluate": lambda a: run([sys.executable, "evaluate.py", *a]),
    "calibrate": lambda a: run([sys.executable, "calibrate.py", *a]),
    "app": app,
    "injector-demo": injector_demo,
}
OTHER_COMMANDS = {"setup": setup, "docker": docker, "docker-stop": docker_stop}


def using_venv():
    return Path(sys.prefix).resolve() == VENV.resolve()


def packages_missing():
    import importlib.util
    return [m for m in NEEDED if importlib.util.find_spec(m) is None]


def main(argv):
    if not argv or argv[0] in ("help", "-h", "--help"):
        say(__doc__)
        return 0
    cmd, args = argv[0], list(argv[1:])
    if cmd in OTHER_COMMANDS:
        return OTHER_COMMANDS[cmd](args)
    if cmd not in PYTHON_COMMANDS:
        say(f"Unknown command '{cmd}'.\n{__doc__}")
        return 2
    if not using_venv() and VENV_PY.exists():
        # use the project's own environment, no activation needed
        return run([VENV_PY, Path(__file__).resolve(), *argv])
    missing = packages_missing()
    if missing:
        say(f"Some packages are missing ({', '.join(missing)}). Install them first with:")
        say(f"  {PY_CMD} run.py setup")
        return 1
    return PYTHON_COMMANDS[cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
