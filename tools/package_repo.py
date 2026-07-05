#!/usr/bin/env python3
"""package_repo.py - build a clean, GitHub-ready ZIP of the project repository.

Produces a ZIP containing only files relevant to the study. Excludes version
control internals, byte-code and test caches, notebook checkpoints, virtual
environments, editor and OS cruft, log and temp files, and the figure layout
sidecar. The reproducibility records env/pip_freeze.txt, env/system.json and
SHA256SUMS.txt are kept, because they are study artifacts, not environment
junk.

Notebook gate (on by default): notebooks/ must contain at least one .ipynb and
every code cell with real source must have been executed (execution_count set
or outputs present). This exists because notebooks were omitted or shipped
un-run in earlier projects; an un-executed notebook fails packaging.

Usage:
    python3 tools/package_repo.py --project . [--out DIR] [--name NAME]
                                  [--check] [--allow-empty-notebooks]
                                  [--require-clean-git]

--check                 validate only; do not write a ZIP
--allow-empty-notebooks skip the notebook gate (discouraged)
--require-clean-git     fail if the Git working tree has uncommitted changes

Stdlib only. Exit 0 on success, 1 on any failure.
"""
import argparse
import datetime
import json
import subprocess
import sys
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    ".mypy_cache", ".ruff_cache", ".venv", "venv", "env-venv", "node_modules",
    ".idea", ".vscode", ".DS_Store", "_layout",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".swp", ".bak"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db", "SHA256SUMS.txt.tmp"}


def excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if rel.name in EXCLUDE_NAMES:
        return True
    if rel.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False


def iter_included(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.name == "SHA256SUMS.txt" and len(rel.parts) == 1:
            yield rel  # keep checksums at root
            continue
        if excluded(rel):
            continue
        yield rel


def notebook_status(nb_path: Path):
    """Return (ok, message). ok=False if any non-empty code cell is un-run."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, f"{nb_path.name}: not valid JSON ({e})"
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"
                  and "".join(c.get("source", [])).strip()]
    if not code_cells:
        return True, f"{nb_path.name}: no executable code cells"
    unrun = []
    for i, c in enumerate(code_cells):
        executed = c.get("execution_count") is not None or bool(c.get("outputs"))
        if not executed:
            unrun.append(i + 1)
    if unrun:
        return False, (f"{nb_path.name}: {len(unrun)} code cell(s) not executed "
                       f"(e.g. cell #{unrun[0]}); run top-to-bottom and re-save")
    return True, f"{nb_path.name}: executed ({len(code_cells)} code cells)"


def check_notebooks(root: Path):
    nbdir = root / "notebooks"
    nbs = sorted(nbdir.glob("*.ipynb")) if nbdir.is_dir() else []
    nbs = [n for n in nbs if ".ipynb_checkpoints" not in n.parts]
    if not nbs:
        return False, ["notebooks/ contains no .ipynb files; at least one "
                       "executed notebook reproducing the results is required"]
    msgs, ok_all = [], True
    for nb in nbs:
        ok, msg = notebook_status(nb)
        ok_all = ok_all and ok
        msgs.append(("OK  " if ok else "FAIL") + " " + msg)
    return ok_all, msgs


def git_is_clean(root: Path) -> bool:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                       capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default="/mnt/user-data/outputs",
                    help="directory to write the ZIP into")
    ap.add_argument("--name", default=None,
                    help="archive base name (default: project directory name)")
    ap.add_argument("--check", action="store_true", help="validate only")
    ap.add_argument("--allow-empty-notebooks", action="store_true")
    ap.add_argument("--require-clean-git", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.project).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return 1

    failures = 0

    if args.require_clean_git:
        if git_is_clean(root):
            print("[PASS] git working tree clean")
        else:
            print("[FAIL] git working tree has uncommitted changes; commit first")
            failures += 1

    if not args.allow_empty_notebooks:
        ok, msgs = check_notebooks(root)
        for m in msgs:
            print(f"[notebook] {m}")
        if not ok:
            print("[FAIL] notebook gate")
            failures += 1
        else:
            print("[PASS] notebook gate")
    else:
        print("[skip] notebook gate (allow-empty-notebooks)")

    included = list(iter_included(root))
    total_bytes = sum((root / r).stat().st_size for r in included)
    print(f"[info] {len(included)} files selected "
          f"({total_bytes / 1024:.1f} KiB) after exclusions")

    if failures:
        print(f"\nPACKAGE REPO: FAIL ({failures} problem(s)); no ZIP written")
        return 1

    if args.check:
        print("\nPACKAGE REPO: CHECK PASS (no ZIP written)")
        return 0

    name = args.name or root.name
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in included:
            zf.write(root / rel, arcname=str(Path(name) / rel))
        manifest = "\n".join(str(Path(name) / r) for r in included) + "\n"
        zf.writestr(str(Path(name) / "PACKAGE_CONTENTS.txt"), manifest)

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"\nPACKAGE REPO: PASS")
    print(f"  wrote {zip_path} ({zip_path.stat().st_size / 1024:.1f} KiB, "
          f"{len(included)} files) at {stamp}")
    print("  top-level folder in the archive: " + name + "/ "
          "(extracts clean for GitHub upload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
