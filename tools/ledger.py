#!/usr/bin/env python3
"""Append-only audit ledger for the autonomous research lab.

Every gate decision, amendment, override request, and STOP is recorded as one
JSON line in <project>/audit/ledger.jsonl. Lines are never edited or deleted;
corrections are new lines referencing the corrected entry.

Usage:
  python3 ledger.py --project . add --phase P2 --gate G2 --decision PASS \
      --evidence audit/novelty_audit.md --evidence docs/GAP_STATEMENT.md \
      --notes "Novelty risk LOW; feasibility verified"
  python3 ledger.py --project . list
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

DECISIONS = ["PASS", "FAIL", "STOP", "WARN", "INFO"]


def ledger_path(project: Path) -> Path:
    return project / "audit" / "ledger.jsonl"


def cmd_add(args) -> int:
    project = Path(args.project).resolve()
    path = ledger_path(project)
    if not path.parent.is_dir():
        print(f"ERROR: {path.parent} does not exist. Run init_project.py first.",
              file=sys.stderr)
        return 2
    missing = [e for e in (args.evidence or []) if not (project / e).exists()]
    if missing:
        print("ERROR: evidence paths do not exist (relative to project root): "
              + ", ".join(missing), file=sys.stderr)
        print("The ledger records evidence, it does not promise it.", file=sys.stderr)
        return 2
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "phase": args.phase,
        "gate": args.gate or "",
        "decision": args.decision,
        "evidence": args.evidence or [],
        "notes": args.notes or "",
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"LEDGER {entry['decision']} {entry['phase']}/{entry['gate']} recorded -> {path}")
    return 0


def cmd_list(args) -> int:
    path = ledger_path(Path(args.project).resolve())
    if not path.exists():
        print("(empty ledger: no entries yet)")
        return 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            print(f"{i:>3}  [UNPARSEABLE LINE - investigate, do not delete]")
            continue
        ev = "; ".join(e.get("evidence", []))
        print(f"{i:>3}  {e.get('ts','')}  {e.get('phase',''):<4} {e.get('gate',''):<4} "
              f"{e.get('decision',''):<5} {e.get('notes','')}" + (f"  [{ev}]" if ev else ""))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=".", help="project root (default: .)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="append a ledger entry")
    a.add_argument("--phase", required=True, help="e.g. P3")
    a.add_argument("--gate", default="", help="e.g. G3")
    a.add_argument("--decision", required=True, choices=DECISIONS)
    a.add_argument("--evidence", action="append",
                   help="repo-relative evidence path (repeatable); must exist")
    a.add_argument("--notes", default="")
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="print the ledger")
    l.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
