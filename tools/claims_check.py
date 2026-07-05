#!/usr/bin/env python3
"""Claim-to-evidence and citation verification for manuscript drafts.

Scans manuscript/draft/*.md for:
  - anchors {{Rnnn}}  -> must resolve to results/register.csv rows with
                         origin_class in A-D and validation_status PASS
  - citations [@bibkey] -> must resolve to literature/matrix.csv rows with
                         verified == yes
ERRORS (exit 1): unknown anchor, anchor to class E, anchor to non-PASS result,
unknown bibkey, unverified bibkey.
WARNINGS (exit 0, listed): quantitative-looking values without an anchor on the
same line in files whose name contains 'result' or 'discussion' or 'abstract'.
Every warning must be dispositioned in audit/claims_warnings.md (anchored, or
justified, e.g. equation numbers and years).

--strip-to DIR writes anchor-free copies for rendering.

Usage:
  python3 claims_check.py --project .
  python3 claims_check.py --project . --strip-to manuscript/clean
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ANCHOR_RE = re.compile(r"\{\{(R\d{3})\}\}")
CITE_RE = re.compile(r"\[@([A-Za-z0-9_\-]+)\]")
# decimals, or integers glued to a unit token: heuristic for quantitative claims
NUM_RE = re.compile(
    r"(?<![\w.])\d+\.\d+(?:[eE][+-]?\d+)?"
    r"|(?<![\w.])\d+(?:\.\d+)?\s?(?:%|nm|um|mm|cm|m|eV|meV|K|s|ms|ns|fs|Hz|kHz|MHz|GHz|THz|dB|deg|rad|W|mW|J|V|A|T|Pa)\b")


def load_register(path: Path):
    reg = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = (row.get("result_id") or "").strip()
                if rid:
                    reg[rid] = row
    return reg


def load_matrix(path: Path):
    mat = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("bibkey") or "").strip()
                if key:
                    mat[key] = row
    return mat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=".")
    ap.add_argument("--manuscript-dir", default="manuscript/draft")
    ap.add_argument("--register", default="results/register.csv")
    ap.add_argument("--literature", default="literature/matrix.csv")
    ap.add_argument("--strip-to", default=None,
                    help="write anchor-stripped copies of the drafts here")
    args = ap.parse_args(argv)

    project = Path(args.project).resolve()
    draft_dir = project / args.manuscript_dir
    register = load_register(project / args.register)
    matrix = load_matrix(project / args.literature)

    errors, warnings = [], []
    md_files = sorted(draft_dir.glob("*.md")) if draft_dir.is_dir() else []
    if not md_files:
        errors.append(f"no draft files found in {draft_dir}")

    for path in md_files:
        warn_eligible = any(k in path.name.lower()
                            for k in ("result", "discussion", "abstract"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            loc = f"{path.name}:{lineno}"
            anchors = ANCHOR_RE.findall(line)
            for rid in anchors:
                row = register.get(rid)
                if row is None:
                    errors.append(f"{loc}: anchor {{{{{rid}}}}} not in register")
                    continue
                cls = (row.get("origin_class") or "").strip().upper()
                status = (row.get("validation_status") or "").strip().upper()
                if cls not in ("A", "B", "C", "D"):
                    errors.append(f"{loc}: {rid} has origin_class '{cls}' "
                                  "(class E / unknown is barred from deliverables)")
                if status != "PASS":
                    errors.append(f"{loc}: {rid} validation_status is "
                                  f"'{status or 'EMPTY'}', not PASS")
            for key in CITE_RE.findall(line):
                row = matrix.get(key)
                if row is None:
                    errors.append(f"{loc}: citation [@{key}] not in literature matrix")
                elif (row.get("verified") or "").strip().lower() != "yes":
                    errors.append(f"{loc}: citation [@{key}] is not verified=yes")
            if warn_eligible and not anchors and not line.lstrip().startswith("#"):
                stripped = CITE_RE.sub("", line)
                if NUM_RE.search(stripped):
                    warnings.append(f"{loc}: quantitative value without an anchor: "
                                    f"{line.strip()[:90]}")

    if args.strip_to:
        out_dir = project / args.strip_to
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in md_files:
            clean = ANCHOR_RE.sub("", path.read_text(encoding="utf-8"))
            clean = re.sub(r"[ \t]+(?=[\s.,;:)\]])", lambda m: "", clean)
            (out_dir / path.name).write_text(clean, encoding="utf-8")
        print(f"stripped copies written to {out_dir}")

    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    print(f"\nCLAIMS CHECK: {len(errors)} error(s), {len(warnings)} warning(s) "
          f"across {len(md_files)} file(s)")
    if warnings and not errors:
        print("Disposition every warning in audit/claims_warnings.md before rendering.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
