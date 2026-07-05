#!/usr/bin/env python3
"""Journal figure style layer for the autonomous research lab.

Enforces physical geometry (single column 9.0 cm, double column 18.0 cm by
default) and journal typography (labels 9 pt, ticks/legend 8 pt) via rcParams,
and saves every figure as a PDF (vector, submission) + PNG (600 dpi, QC and
docx embedding) pair. Constrained layout fits labels inside a FIXED canvas, so
the saved physical width is exact and figure_qc.py can verify it; never save
with bbox_inches='tight', which silently resizes the canvas.

save() also runs a renderer-based LAYOUT CHECK: it draws the figure, measures
the bounding boxes of titles, axis labels, tick-label groups, legends, and
panel labels, and reports overlaps, intrusions into a plotting region, and
labels clipped at the figure edge. The findings are written to
figures/_layout/<name>.json; figure_qc.py fails any figure whose report is
missing or non-empty. This replaces the old "overlap is guaranteed by
construction" assumption with an executed check.

Usage in a figure script:
    import fig_style
    fig, ax = fig_style.new_figure(kind="single", height_cm=7.0)
    ...plot...
    fig_style.legend_outside(ax)
    fig_style.save(fig, "figures/fig1_name")
"""
from __future__ import annotations

CM = 1.0 / 2.54
SINGLE_W_CM = 9.0    # journal band 8-10 cm
DOUBLE_W_CM = 18.0   # journal band 16-20 cm
PNG_DPI = 600        # >= 300 required by QC

# Layout-check thresholds (display pixels at the figure's on-screen dpi).
_OVERLAP_MIN_AREA = 4.0     # ignore hairline touches below this area
_OVERLAP_MIN_SIDE = 1.0     # require >1 px overlap on both axes
_INTRUDE_FRAC = 0.15        # label area fraction inside a foreign plotting region
_EDGE_TOL = 2.0             # px tolerance before a label counts as clipped

_BASE_RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "DejaVu Sans",
    "axes.labelpad": 4.0,         # keep axis titles off the data region
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.045,   # inches, inter-element padding
    "figure.constrained_layout.w_pad": 0.045,
    "figure.constrained_layout.hspace": 0.06,   # fraction, inter-panel space
    "figure.constrained_layout.wspace": 0.06,
    "savefig.dpi": PNG_DPI,
    "savefig.bbox": "standard",   # keep the physical canvas size exact
    "pdf.fonttype": 42,           # embed TrueType (journal requirement)
    "ps.fonttype": 42,
    "axes.prop_cycle": None,      # set in apply_style (needs matplotlib)
}


def apply_style(width_cm: float = SINGLE_W_CM, height_cm: float = 7.0):
    """Set rcParams for one figure size. Called by new_figure."""
    import matplotlib
    from cycler import cycler
    rc = dict(_BASE_RC)
    rc["figure.figsize"] = (width_cm * CM, height_cm * CM)
    # tab10: colorblind-reasonable default; pair with linestyle/marker variation.
    rc["axes.prop_cycle"] = cycler(color=matplotlib.colormaps["tab10"].colors)
    matplotlib.rcParams.update(rc)


def new_figure(kind: str = "single", height_cm: float | None = None,
               width_cm: float | None = None, **subplots_kw):
    """Create a styled (fig, ax(es)). kind: 'single' or 'double'.

    Accepts the usual plt.subplots keywords (nrows, ncols, sharex, ...).
    """
    if kind not in ("single", "double"):
        raise ValueError("kind must be 'single' or 'double'")
    w = width_cm if width_cm is not None else (SINGLE_W_CM if kind == "single" else DOUBLE_W_CM)
    lo, hi = (8.0, 10.0) if kind == "single" else (16.0, 20.0)
    if not (lo <= w <= hi):
        raise ValueError(f"{kind}-column width must be {lo}-{hi} cm, got {w}")
    h = height_cm if height_cm is not None else round(w / 1.45, 2)
    apply_style(w, h)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(**subplots_kw)
    fig._arl_kind = kind          # consumed by save() for the manifest hint
    fig._arl_width_cm = w
    return fig, ax


def legend_outside(ax, loc: str = "upper left"):
    """Place the legend outside the axes (right side) without covering data."""
    return ax.legend(loc=loc, bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def panel_label(ax, text: str, loc: str = "upper left"):
    """Add a bold panel label (a), (b), ... tagged for the layout check.

    Sits in the top corner of the axes with a light background so it reads
    above the data. Tagged with _arl_panel so save() can reason about it.
    """
    pos = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }.get(loc, (0.02, 0.98, "left", "top"))
    x, y, ha, va = pos
    t = ax.text(x, y, text, transform=ax.transAxes, fontsize=9,
                fontweight="bold", ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))
    t._arl_panel = True
    return t


# --------------------------------------------------------------------------- #
# Layout validation
# --------------------------------------------------------------------------- #
def _union(bboxes):
    from matplotlib.transforms import Bbox
    xs0 = min(b.x0 for b in bboxes); ys0 = min(b.y0 for b in bboxes)
    xs1 = max(b.x1 for b in bboxes); ys1 = max(b.y1 for b in bboxes)
    return Bbox([[xs0, ys0], [xs1, ys1]])


def _overlap(a, b):
    """Return (overlap_w, overlap_h, overlap_area) of two display Bboxes."""
    x0 = max(a.x0, b.x0); y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1); y1 = min(a.y1, b.y1)
    w = x1 - x0; h = y1 - y0
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0
    return w, h, w * h


def _tick_group(ax, axis, renderer):
    """Union bbox of the in-range, visible tick labels on one axis, or None.

    Off-range tick labels (which matplotlib keeps but clips) are excluded, so
    the union does not spill past the axes and trigger false clipping/overlap.
    """
    if axis == "x":
        ticks, lim, labels = ax.get_xticks(), ax.get_xlim(), ax.get_xticklabels()
    else:
        ticks, lim, labels = ax.get_yticks(), ax.get_ylim(), ax.get_yticklabels()
    lo, hi = (min(lim), max(lim))
    boxes = []
    for t, lab in zip(ticks, labels):
        try:
            if (lab.get_visible() and str(lab.get_text()).strip()
                    and lo - 1e-9 <= t <= hi + 1e-9):
                boxes.append(lab.get_window_extent(renderer=renderer))
        except Exception:
            pass
    return _union(boxes) if boxes else None


def _gather(fig, renderer):
    """Return (major_labels, tick_groups).

    major_labels: discrete text elements (suptitle, titles, axis labels,
    legends, panel labels) that should never collide. tick_groups: per-axis
    in-range tick-label unions, compared only across different axes (within one
    axes the x and y groups meet at the corner by construction).
    """
    major, ticks = [], []
    axes = list(fig.axes)

    def add(artist, label, role, ax_index):
        if artist is None:
            return
        try:
            if not artist.get_visible():
                return
            if hasattr(artist, "get_text") and not str(artist.get_text()).strip():
                return
            bb = artist.get_window_extent(renderer=renderer)
        except Exception:
            return
        if bb.width > 0 and bb.height > 0:
            major.append((label, role, ax_index, bb))

    add(getattr(fig, "_suptitle", None), "suptitle", "suptitle", None)
    for i, ax in enumerate(axes):
        if not ax.get_visible():
            continue
        add(ax.title, f"axes[{i}] title", "title", i)
        add(ax.xaxis.label, f"axes[{i}] x-label", "xlabel", i)
        add(ax.yaxis.label, f"axes[{i}] y-label", "ylabel", i)
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            try:
                major.append((f"axes[{i}] legend", "legend", i,
                              leg.get_window_extent(renderer=renderer)))
            except Exception:
                pass
        for t in ax.texts:
            if getattr(t, "_arl_panel", False):
                add(t, f"axes[{i}] panel label '{t.get_text()}'", "panel", i)
        for axis in ("x", "y"):
            g = _tick_group(ax, axis, renderer)
            if g is not None:
                ticks.append((f"axes[{i}] {axis}-ticklabels", f"{axis}ticks", i, g))

    for leg in getattr(fig, "legends", []):
        try:
            if leg.get_visible():
                major.append(("figure legend", "legend", None,
                              leg.get_window_extent(renderer=renderer)))
        except Exception:
            pass
    return major, ticks


def check_layout(fig):
    """Draw the figure and return a list of human-readable layout problems."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W = float(fig.bbox.width); H = float(fig.bbox.height)
    major, ticks = _gather(fig, renderer)
    problems = []

    def clipped(bb):
        return (bb.x0 < -_EDGE_TOL or bb.y0 < -_EDGE_TOL
                or bb.x1 > W + _EDGE_TOL or bb.y1 > H + _EDGE_TOL)

    def overlaps(a, b):
        ow, oh, oa = _overlap(a, b)
        return oa >= _OVERLAP_MIN_AREA and ow > _OVERLAP_MIN_SIDE and oh > _OVERLAP_MIN_SIDE

    # 1. Clipping at the figure edge (discrete labels and in-range tick groups).
    for label, _role, _ai, bb in major + ticks:
        if clipped(bb):
            problems.append(f"{label} is clipped at the figure edge")

    # 2. Overlap among discrete labels anywhere in the figure.
    for a in range(len(major)):
        la, ra, ia, ba = major[a]
        for b in range(a + 1, len(major)):
            lb, rb, ib, bb = major[b]
            if ("panel" in (ra, rb)) and ia is not None and ia == ib:
                continue  # a panel label may sit near its own axis content
            if overlaps(ba, bb):
                problems.append(f"{la} overlaps {lb}")

    # 3. Tick groups vs labels and tick groups of *other* axes only.
    for lt, _rt, it, bt in ticks:
        for lm, _rm, im, bm in major:
            if im == it:
                continue
            if overlaps(bt, bm):
                problems.append(f"{lt} overlaps {lm}")
    for a in range(len(ticks)):
        lt, _ra, ia, ba = ticks[a]
        for b in range(a + 1, len(ticks)):
            lu, _rb, ib, bb = ticks[b]
            if ia == ib:
                continue  # x and y groups of one axes meet at the corner
            if overlaps(ba, bb):
                problems.append(f"{lt} overlaps {lu}")

    # 4. A discrete label intruding into a *different* axes' plotting region.
    data_boxes = {}
    for i, ax in enumerate(fig.axes):
        try:
            if ax.get_visible():
                data_boxes[i] = ax.get_window_extent(renderer=renderer)
        except Exception:
            pass
    for label, role, ai, bb in major:
        if role not in ("title", "xlabel", "ylabel", "legend", "panel"):
            continue
        area = bb.width * bb.height
        if area <= 0:
            continue
        for j, dbox in data_boxes.items():
            if j == ai:
                continue
            _ow, _oh, oa = _overlap(bb, dbox)
            if oa / area > _INTRUDE_FRAC:
                problems.append(f"{label} intrudes into the plotting region of axes[{j}]")

    seen, unique = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p); unique.append(p)
    return unique


def _write_layout_report(figdir, name, problems):
    import json
    from pathlib import Path
    ldir = Path(figdir) / "_layout"
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / f"{name}.json").write_text(
        json.dumps({"figure": name, "n_problems": len(problems),
                    "problems": problems}, indent=2) + "\n",
        encoding="utf-8")


def save(fig, basepath: str, check: bool = True):
    """Save <basepath>.pdf and <basepath>.png; returns the two paths.

    When check is True (default), run the layout validation and write the
    report next to the figure so figure_qc.py can enforce it. A figure with
    layout problems is still written (for inspection) but will fail QC.
    """
    from pathlib import Path
    base = Path(basepath)
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")

    problems = []
    if check:
        try:
            problems = check_layout(fig)
        except Exception as e:  # never let the check crash a run
            problems = [f"layout check could not run: {e}"]
        _write_layout_report(base.parent, base.stem, problems)

    fig.savefig(pdf)                  # vector, fonttype 42 via rcParams
    fig.savefig(png, dpi=PNG_DPI)     # raster with dpi metadata for QC

    status = "OK" if not problems else f"{len(problems)} LAYOUT PROBLEM(S)"
    print(f"saved {pdf} and {png} "
          f"({getattr(fig, '_arl_width_cm', '?')} cm, "
          f"{getattr(fig, '_arl_kind', '?')} column) [{status}]")
    for p in problems:
        print(f"  LAYOUT: {p}")
    return str(pdf), str(png)
