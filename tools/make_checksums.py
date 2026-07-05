#!/usr/bin/env python3
"""make_checksums.py - write or verify a SHA-256 manifest over a project tree.

Write mode (default): hashes every file under --project except .git/ and the
manifest itself, writes SHA256SUMS.txt at the project root (sorted by path).

Verify mode (--verify): recomputes hashes and compares against SHA256SUMS.txt.
Exit 1 on any mismatch, missing file, or file present on disk but absent from
the manifest (drift detection cuts both ways).

Usage:
    python3 tools/make_checksums.py --project .
    python3 tools/make_checksums.py --project . --verify

Stdlib only.
"""
import argparse
import hashlib
import sys
from pathlib import Path

MANIFEST = "SHA256SUMS.txt"
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.name == MANIFEST and len(rel.parts) == 1:
            continue
        yield rel


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path) -> int:
    lines = []
    for rel in iter_files(root):
        digest = sha256_of(root / rel)
        lines.append(f"{digest}  {rel.as_posix()}")
    (root / MANIFEST).write_text("\n".join(lines) + "\n")
    print(f"Wrote {MANIFEST}: {len(lines)} files hashed.")
    return 0


def verify_manifest(root: Path) -> int:
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        print(f"FAIL: {MANIFEST} not found at {root}")
        return 1
    expected = {}
    for line in manifest_path.read_text().splitlines():
        line = line.rstrip()
        if not line:
            continue
        try:
            digest, rel = line.split(None, 1)
        except ValueError:
            print(f"FAIL: malformed manifest line: {line!r}")
            return 1
        expected[rel.strip()] = digest

    failures = []
    on_disk = {rel.as_posix() for rel in iter_files(root)}

    for rel, digest in sorted(expected.items()):
        p = root / rel
        if not p.exists():
            failures.append(f"MISSING  {rel}")
            continue
        actual = sha256_of(p)
        if actual != digest:
            failures.append(f"MISMATCH {rel}")

    for rel in sorted(on_disk - set(expected)):
        failures.append(f"UNTRACKED {rel} (on disk, not in manifest)")

    if failures:
        print(f"VERIFY FAIL: {len(failures)} problem(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"VERIFY PASS: {len(expected)} files match.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    root = Path(args.project).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        sys.exit(1)

    sys.exit(verify_manifest(root) if args.verify else write_manifest(root))


if __name__ == "__main__":
    main()
