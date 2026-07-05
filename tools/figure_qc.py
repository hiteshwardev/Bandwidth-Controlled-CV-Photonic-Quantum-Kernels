#!/usr/bin/env python3
"""Automated figure quality audit (stdlib only).

Checks every row of figures/manifest.csv:
  - basename.png and basename.pdf exist and are non-empty
  - PNG carries dpi metadata (pHYs chunk) with dpi >= 300
  - physical width = pixels/dpi lies in the journal band
        single: 8-10 cm    double: 16-20 cm
  - the listed source script and data paths exist
  - caption_stub is non-empty
  - a layout report (figures/_layout/<name>.json, written by fig_style.save)
    exists and reports zero overlaps, intrusions, or clipped labels
and fails on ORPHANS: any .png/.pdf in figures/ not listed in the manifest,
because an untracked figure is an unprovenanced figure.

Exit 0 only if every check passes. Report: figures/qc_report.txt.

Usage: python3 figure_qc.py --project .
"""
import argparse
import csv
import json
import struct
import sys
from pathlib import Path

BANDS = {"single": (8.0, 10.0), "double": (16.0, 20.0)}
SLACK_CM = 0.05
MIN_DPI = 300.0


def png_geometry(path: Path):
    """Return (width_px, height_px, dpi_x or None) from IHDR / pHYs chunks."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos, width, height, dpi = 8, None, None, None
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif ctype == b"pHYs" and length >= 9:
            ppux, _ppuy, unit = struct.unpack(">IIB", body[:9])
            if unit == 1 and ppux > 0:           # pixels per metre
                dpi = ppux * 0.0254
        elif ctype == b"IDAT":
            break
        pos += 12 + length
    return width, height, dpi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=".")
    ap.add_argument("--manifest", default="figures/manifest.csv")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args(argv)

    project = Path(args.project).resolve()
    manifest = project / args.manifest
    figdir = project / args.figdir
    lines, failures = [], 0

    def check(ok: bool, msg: str):
        nonlocal failures
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        lines.append(f"[{tag}] {msg}")

    if not manifest.exists():
        check(False, f"manifest missing: {manifest}")
        rows = []
    else:
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)]
        check(bool(rows), f"manifest has at least one figure ({len(rows)} found)")

    listed = set()
    for r in rows:
        name = (r.get("filename") or "").strip()
        kind = (r.get("kind") or "").strip().lower()
        if not name:
            check(False, "manifest row with empty filename")
            continue
        # Normalize: manifest may list the basename or the .png/.pdf filename
        if name.lower().endswith((".png", ".pdf")):
            name = name[:-4]
        listed.add(name)
        png, pdf = figdir / f"{name}.png", figdir / f"{name}.pdf"
        check(png.exists() and png.stat().st_size > 0, f"{name}.png exists and non-empty")
        check(pdf.exists() and pdf.stat().st_size > 0, f"{name}.pdf exists and non-empty")
        check(kind in BANDS, f"{name}: kind '{kind}' is single|double")

        if png.exists() and kind in BANDS:
            try:
                w_px, _h_px, dpi = png_geometry(png)
            except Exception as e:  # noqa: BLE001
                check(False, f"{name}.png unreadable as PNG ({e})")
            else:
                if dpi is None:
                    check(False, f"{name}.png has no dpi metadata (pHYs chunk); "
                                 "save through fig_style.save()")
                else:
                    check(dpi >= MIN_DPI - 0.5,
                          f"{name}.png dpi {dpi:.0f} >= {MIN_DPI:.0f}")
                    width_cm = w_px / dpi * 2.54
                    lo, hi = BANDS[kind]
                    check(lo - SLACK_CM <= width_cm <= hi + SLACK_CM,
                          f"{name}.png physical width {width_cm:.2f} cm in "
                          f"{kind} band [{lo}, {hi}] cm")

        for field in ("script", "data"):
            val = (r.get(field) or "").strip()
            if field == "data" and val.upper() == "NA":
                lines.append(f"[PASS] {name}: data declared NA (analytic figure)")
                continue
            check(bool(val) and (project / val).exists(),
                  f"{name}: {field} path exists ({val or 'EMPTY'})")
        check(bool((r.get("caption_stub") or "").strip()),
              f"{name}: caption_stub non-empty")

        # Layout report written by fig_style.save(): must exist and be clean.
        layout = figdir / "_layout" / f"{name}.json"
        if not layout.exists():
            check(False, f"{name}: layout report missing "
                         "(save through fig_style.save())")
        else:
            try:
                rep = json.loads(layout.read_text(encoding="utf-8"))
                nprob = int(rep.get("n_problems", -1))
            except Exception as e:  # noqa: BLE001
                check(False, f"{name}: layout report unreadable ({e})")
            else:
                if nprob == 0:
                    lines.append(f"[PASS] {name}: layout clean (no overlaps/clipping)")
                else:
                    probs = rep.get("problems", [])
                    check(False, f"{name}: {nprob} layout problem(s): "
                                 + "; ".join(probs[:4])
                                 + (" ..." if len(probs) > 4 else ""))

    if figdir.is_dir():
        for f in sorted(figdir.iterdir()):
            if f.suffix.lower() in (".png", ".pdf") and f.stem not in listed:
                check(False, f"ORPHAN figure not in manifest: {f.name}")

    verdict = "FIGURE QC: PASS" if failures == 0 else f"FIGURE QC: FAIL ({failures} failure(s))"
    report = "\n".join(lines + ["", verdict]) + "\n"
    if figdir.is_dir():
        (figdir / "qc_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
