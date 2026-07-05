#!/usr/bin/env python3
"""init_project.py - scaffold an Autonomous Research Lab project repository.

Creates the full directory tree defined in references/repository-standards.md,
initializes Git, writes the audit ledger's first entry, creates the results
register / literature matrix / figure manifest headers, drops a LOCKED:NO
pre-registration template, and copies the sibling lab scripts into the
project's tools/ directory so the shipped repository is self-contained.

Usage (run from the skill directory):
    python3 scripts/init_project.py <project_dir> --title "Project title"
        [--authors "..."] [--affiliation "..."] [--no-env-audit]

Stdlib only.
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_AUTHORS = "Hitesh Kumar Singh (first, corresponding); Mr. Joh"
DEFAULT_AFFILIATION = "MNS Government College, Bhiwani"

TOOL_FILES = [
    "ledger.py",
    "env_audit.py",
    "fig_style.py",
    "figure_qc.py",
    "claims_check.py",
    "make_checksums.py",
    "package_repo.py",
    "init_project.py",
]

DIRS = [
    "data/raw",
    "data/processed",
    "literature",
    "notebooks",
    "src",
    "tests",
    "results",
    "figures",
    "manuscript/draft",
    "manuscript/clean",
    "audit",
    "docs",
    "tools",
    "env",
]

GITIGNORE = """__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
*.egg-info/
"""

REGISTER_HEADER = (
    "result_id,description,origin_class,evidence_path,verification_method,"
    "validation_status,confidence,uncertainty,reproducibility,notes\n"
)

MATRIX_HEADER = (
    "bibkey,authors,year,title,venue,doi_or_url,method,main_finding,"
    "limitation,verified,verification_evidence\n"
)

MANIFEST_HEADER = "filename,kind,script,data,caption_stub\n"

PREREG_TEMPLATE = """# Pre-Registration

LOCKED: NO
LOCK_COMMIT: (git hash recorded at lock time)
LOCK_DATE: (date recorded at lock time)

Once LOCKED is set to YES this file is frozen. Changes after locking are
permitted only as numbered, dated entries in the Amendments section below,
each with a stated reason. Editing locked sections is an integrity violation.

## 1. Research question

(state the question precisely)

## 2. Primary hypothesis and falsification criterion

(H1, the observable that would falsify it, and the decision threshold)

## 3. Secondary hypotheses

(up to three, each with falsification criteria)

## 4. Success criteria

(quantitative criteria fixed before any production run)

## 5. Validation benchmarks

(the analytical and literature benchmarks the results must reproduce,
with tolerances)

## 6. Statistical plan

(uncertainty quantification methods, convergence criteria, sensitivity
parameters and ranges, error estimation approach)

## 7. Planned runs

(benchmark runs first, then production runs; parameters and expected
runtime class for each)

## 8. Stopping rules

(conditions under which the project halts or descopes)

## Amendments

(none)
"""

EXECUTION_ORDER_SKELETON = """# Execution Order

Numbered commands that take a fresh clone to the full result set. Every line
must be a command that was actually executed in this project (distilled from
run records and the audit ledger), in dependency order.

1. pip install -r requirements.txt --break-system-packages
2. python3 -m pytest tests/ -v
(extend as the project executes; do not list aspirational commands)
"""

LICENSE_PLACEHOLDER = """TODO: confirm license choice with the project owner at Phase 7.
Default proposal: MIT License. Replace this file with the chosen license's
full text before packaging. Do not publish with this placeholder in place.
"""

CONDUCT_NOTE = """# Audit Directory

Append-only evidence. Files here are never edited after being written;
superseded reports get a timestamp suffix, they are not overwritten.
"""


def run(cmd, cwd):
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_dir", help="Directory to create (must not exist or be empty)")
    ap.add_argument("--title", required=True, help="Project title")
    ap.add_argument("--authors", default=DEFAULT_AUTHORS)
    ap.add_argument("--affiliation", default=DEFAULT_AFFILIATION)
    ap.add_argument("--no-env-audit", action="store_true",
                    help="Skip the initial environment audit (not recommended)")
    args = ap.parse_args()

    root = Path(args.project_dir).resolve()
    if root.exists() and any(root.iterdir()):
        print(f"ERROR: {root} exists and is not empty. Refusing to scaffold over it.")
        sys.exit(1)
    root.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today().isoformat()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

    # Core metadata files
    (root / ".gitignore").write_text(GITIGNORE)
    (root / "LICENSE").write_text(LICENSE_PLACEHOLDER)
    (root / "audit" / "README.md").write_text(CONDUCT_NOTE)
    (root / "notebooks" / "README.md").write_text(
        "# Notebooks\n\n"
        "Every notebook here mirrors code in src/ and must be executed "
        "top-to-bottom before packaging (no stripped or un-run cells). The set "
        "of notebooks must reproduce the reported results, regenerate every "
        "figure used in the manuscript, and match the outputs produced in the "
        "working environment. tools/package_repo.py enforces that notebooks are "
        "present and executed before it writes the GitHub ZIP.\n"
    )

    (root / "README.md").write_text(
        f"# {args.title}\n\n"
        f"Authors: {args.authors}\n"
        f"Affiliation: {args.affiliation}\n"
        f"Scaffolded: {today} by the Autonomous Research Lab skill.\n\n"
        "## Status\n\nPhase 0 (scaffold). Nothing in this repository is validated yet.\n\n"
        "## How to reproduce\n\nSee docs/EXECUTION_ORDER.md (populated as the project executes).\n\n"
        "## Environment\n\nSee env/ and audit/environment_audit.json. "
        "Installs in containerized environments may require "
        "`pip install --break-system-packages`.\n\n"
        "## Repository map\n\nSee docs/ and the project tree in the repository standards.\n\n"
        "## Results-to-claims map\n\nresults/register.csv is the single source of truth; "
        "manuscript claims anchor to register IDs.\n\n"
        "## License and citation\n\nLICENSE (placeholder until Phase 7), CITATION.cff.\n"
    )

    (root / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "message: If you use this work, please cite it.\n"
        f"title: \"{args.title}\"\n"
        "authors:\n"
        "  - family-names: Singh\n"
        "    given-names: Hitesh Kumar\n"
        f"    affiliation: \"{args.affiliation}\"\n"
        "  - name: \"Mr. Joh\"\n"
        f"    affiliation: \"{args.affiliation}\"\n"
        f"date-released: \"{today}\"\n"
    )

    (root / "requirements.txt").write_text(
        "# Pinned at Phase 7 from audit/environment_audit.json (verified versions only).\n"
    )

    # Registers and manifests with exact headers the checking tools parse
    (root / "results" / "register.csv").write_text(REGISTER_HEADER)
    (root / "literature" / "matrix.csv").write_text(MATRIX_HEADER)
    (root / "literature" / "search_log.md").write_text(
        f"# Search Log\n\nStarted {today}. Record every query, source, date, and hit count.\n"
    )
    (root / "figures" / "manifest.csv").write_text(MANIFEST_HEADER)

    # Docs skeletons
    (root / "docs" / "PREREGISTRATION.md").write_text(PREREG_TEMPLATE)
    (root / "docs" / "EXECUTION_ORDER.md").write_text(EXECUTION_ORDER_SKELETON)

    # Self-contained tool copies
    src_dir = Path(__file__).resolve().parent
    copied = []
    for name in TOOL_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, root / "tools" / name)
            copied.append(name)
    missing = [n for n in TOOL_FILES if n not in copied]

    # Git init + first commit
    git_ok = False
    r = run(["git", "init", "-q"], root)
    if r.returncode == 0:
        run(["git", "config", "user.email", "lab@autonomous-research"], root)
        run(["git", "config", "user.name", "Autonomous Research Lab"], root)
        run(["git", "add", "-A"], root)
        c = run(["git", "commit", "-q", "-m", "G0: scaffold repository"], root)
        git_ok = c.returncode == 0
        if not git_ok:
            print(f"WARNING: git commit failed: {c.stderr.strip()}")
    else:
        print(f"WARNING: git init failed: {r.stderr.strip()}")

    # Ledger first entry (direct write; ledger.py requires evidence paths to exist,
    # and README.md does)
    entry = {
        "ts": now,
        "phase": "P0",
        "gate": "G0",
        "decision": "INFO",
        "evidence": ["README.md", "docs/PREREGISTRATION.md"],
        "notes": f"Repository scaffolded: '{args.title}'. Tools copied: {', '.join(copied)}."
                 + (f" MISSING TOOLS: {', '.join(missing)}." if missing else ""),
    }
    with open(root / "audit" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Initial environment audit
    if not args.no_env_audit:
        ea = run([sys.executable, "tools/env_audit.py", "--project", "."], root)
        sys.stdout.write(ea.stdout)
        if ea.returncode != 0:
            print(f"WARNING: env_audit failed: {ea.stderr.strip()}")

    print(f"\nScaffolded: {root}")
    print(f"Git initialized: {'yes' if git_ok else 'NO (see warning above)'}")
    if missing:
        print(f"MISSING TOOL COPIES: {missing} (copy manually before shipping)")
    print("Next steps: complete the P0 intake table with the user, then enter P1 "
          "(literature). Lock docs/PREREGISTRATION.md before any production run.")


if __name__ == "__main__":
    main()
