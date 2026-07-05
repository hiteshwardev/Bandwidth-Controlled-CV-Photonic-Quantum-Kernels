#!/usr/bin/env python3
"""Environment and tool capability audit with executed evidence.

Probes packages by actually importing them; optionally installs a package
(pip --break-system-packages) and then re-verifies by import. Writes a JSON
report so later phases cite evidence instead of memory.

Usage:
  python3 env_audit.py --probe numpy,scipy,matplotlib,meep
  python3 env_audit.py --project . --probe numpy,uncertainties
  python3 env_audit.py --project . --install uncertainties
"""
import argparse
import datetime
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

DEFAULT_PROBE = ("numpy,scipy,matplotlib,pandas,sympy,pypdf,PIL,h5py,"
                 "sklearn,statsmodels,uncertainties,networkx")

# pip name -> import name, for the usual offenders
IMPORT_NAME = {
    "pillow": "PIL", "scikit-learn": "sklearn", "opencv-python": "cv2",
    "pyyaml": "yaml", "beautifulsoup4": "bs4", "python-docx": "docx",
}


def probe(name: str) -> dict:
    mod_name = IMPORT_NAME.get(name.lower(), name)
    try:
        mod = importlib.import_module(mod_name)
        return {"status": "AVAILABLE",
                "version": str(getattr(mod, "__version__", "unknown")),
                "import_name": mod_name}
    except Exception as e:  # noqa: BLE001 - any import failure is evidence
        return {"status": "UNAVAILABLE",
                "error": f"{type(e).__name__}: {e}",
                "import_name": mod_name}


def try_install(name: str) -> dict:
    cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-6:])
    record = {"install_command": " ".join(cmd),
              "install_returncode": proc.returncode,
              "install_log_tail": tail}
    if proc.returncode != 0:
        record["status"] = "UNAVAILABLE"
        return record
    # An install is only verified by a successful import in a fresh interpreter.
    check = subprocess.run(
        [sys.executable, "-c",
         f"import importlib; m=importlib.import_module("
         f"'{IMPORT_NAME.get(name.lower(), name)}');"
         f"print(getattr(m,'__version__','unknown'))"],
        capture_output=True, text=True)
    if check.returncode == 0:
        record["status"] = "INSTALLABLE"
        record["version"] = check.stdout.strip()
    else:
        record["status"] = "UNAVAILABLE"
        record["error"] = ("installed but import failed: "
                           + check.stderr.strip().splitlines()[-1:][0] if check.stderr else "unknown")
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=None,
                   help="project root; report written to <project>/audit/environment_audit.json")
    p.add_argument("--probe", default=DEFAULT_PROBE,
                   help="comma-separated package list to import-test")
    p.add_argument("--install", action="append", default=[],
                   help="package to pip-install and verify (repeatable)")
    args = p.parse_args(argv)

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
    }
    for name in [n.strip() for n in args.probe.split(",") if n.strip()]:
        report["packages"][name] = probe(name)
    for name in args.install:
        report["packages"][name] = try_install(name)

    counts = {}
    for rec in report["packages"].values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    report["summary"] = counts

    text = json.dumps(report, indent=2)
    if args.project:
        out = Path(args.project).resolve() / "audit" / "environment_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Keep history: never overwrite evidence silently.
        if out.exists():
            stamp = report["timestamp"].replace(":", "").replace("-", "")
            out.rename(out.with_name(f"environment_audit_{stamp}.prev.json"))
        out.write_text(text, encoding="utf-8")
        print(f"Report written: {out}")
    print(text)
    for name, rec in report["packages"].items():
        print(f"  {rec['status']:<11} {name} {rec.get('version','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
