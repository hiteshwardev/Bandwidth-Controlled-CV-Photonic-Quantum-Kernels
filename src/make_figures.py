"""Generate the figure set from results/*.csv using the project figure-style layer.

Run: python3 src/make_figures.py
Produces PDF+PNG pairs in figures/ and layout reports in figures/_layout/.
"""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import fig_style
import matplotlib.pyplot as plt
import numpy as np

R = pathlib.Path(__file__).resolve().parents[1] / "results"


def _load(name):
    with open(R / name) as f:
        return list(csv.DictReader(f))


def fig1_concentration_vs_M():
    rows = _load("gaussian_concentration.csv")
    series = [(0.7, 1.0), (0.7, 0.5), (0.7, 0.1), (1.5, 1.0)]
    markers = ["o", "s", "^", "D"]
    fig, ax = fig_style.new_figure(kind="single", height_cm=7.0)
    for (r, eta), mk in zip(series, markers):
        pts = [(int(x["M"]), float(x["var_exact"])) for x in rows
               if float(x["r"]) == r and float(x["eta"]) == eta]
        pts.sort()
        M = [p[0] for p in pts]; V = [p[1] for p in pts]
        ax.semilogy(M, V, marker=mk, label=f"r={r}, $\\eta$={eta}")
    ax.set_xlabel("Number of modes $M$")
    ax.set_ylabel(r"$\mathrm{Var}_{x,x'}[k]$")
    fig_style.legend_outside(ax)
    fig_style.save(fig, "figures/fig1_concentration_vs_M")
    plt.close(fig)


def fig2_rate_vs_loss():
    rows = _load("rate_vs_loss.csv")
    fig, ax = fig_style.new_figure(kind="single", height_cm=7.0)
    markers = {0.3: "o", 0.7: "s", 1.0: "^"}
    for r in [0.3, 0.7, 1.0]:
        pts = [(float(x["eta"]), float(x["rate_per_mode"])) for x in rows if float(x["r"]) == r]
        pts.sort()
        eta = [p[0] for p in pts]; c = [p[1] for p in pts]
        ax.plot(eta, c, marker=markers[r], label=f"r={r}")
    ax.set_xlabel(r"Transmissivity $\eta$  (1 = lossless)")
    ax.set_ylabel(r"Per-mode rate $c$ (nats/mode)")
    fig_style.legend_outside(ax)
    fig_style.save(fig, "figures/fig2_rate_vs_loss")
    plt.close(fig)


def fig3_bandwidth_compensation():
    rows = _load("bandwidth_compensation.csv")
    rows = sorted(rows, key=lambda x: float(x["eta"]))
    eta = [float(x["eta"]) for x in rows]
    disp = [float(x["disp_needed"]) for x in rows]
    fig, ax = fig_style.new_figure(kind="single", height_cm=7.0)
    ax.plot(eta, disp, marker="o", color="C3")
    ax.set_xlabel(r"Transmissivity $\eta$")
    ax.set_ylabel("Displacement scale to\nrestore lossless bandwidth")
    for e, d in zip(eta, disp):
        ax.annotate(f"{d:.2f}", (e, d), textcoords="offset points", xytext=(0, 5), fontsize=7)
    fig_style.save(fig, "figures/fig3_bandwidth_compensation")
    plt.close(fig)


def fig4_gbs_vs_gaussian():
    rows = _load("gbs_concentration.csv")
    fig, axes = fig_style.new_figure(kind="double", height_cm=7.5, nrows=1, ncols=2)
    # panel (a): Var vs M, GBS vs Gaussian, at r=0.6 eta=1.0
    ax = axes[0]
    for r, eta in [(0.6, 1.0)]:
        pts = [(int(x["M"]), float(x["gbs_var"]), float(x["gauss_var_exact"])) for x in rows
               if float(x["r"]) == r and float(x["eta"]) == eta]
        pts.sort()
        M = [p[0] for p in pts]
        ax.semilogy(M, [p[1] for p in pts], marker="o", label="GBS feature kernel")
        ax.semilogy(M, [p[2] for p in pts], marker="s", label="Gaussian kernel")
    ax.set_xlabel("Number of modes $M$")
    ax.set_ylabel(r"$\mathrm{Var}_{x,x'}[k]$")
    ax.set_xticks([1, 2, 3])
    fig_style.legend_outside(ax)
    fig_style.panel_label(ax, "(a)")
    # panel (b): GBS Var vs eta at fixed M=3
    ax2 = axes[1]
    for r, mk in [(0.3, "o"), (0.6, "s")]:
        pts = [(float(x["eta"]), float(x["gbs_var"])) for x in rows
               if float(x["r"]) == r and int(x["M"]) == 3]
        pts.sort()
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, label=f"r={r}")
    ax2.set_xlabel(r"Transmissivity $\eta$")
    ax2.set_ylabel(r"GBS $\mathrm{Var}_{x,x'}[k]$ ($M{=}3$)")
    fig_style.legend_outside(ax2)
    fig_style.panel_label(ax2, "(b)")
    fig_style.save(fig, "figures/fig4_gbs_vs_gaussian")
    plt.close(fig)


def fig5_concentration_vs_simulability():
    rows = _load("h4_boundary.csv")
    fig, axes = fig_style.new_figure(kind="double", height_cm=7.5, nrows=1, ncols=2)
    markers = {0.3:"^", 0.5: "o", 1.0: "s"}
    # panel (a): concentration rate vs eta (all r); positive throughout
    axc = axes[0]
    for r in [0.3, 0.5, 1.0]:
        sub = sorted([x for x in rows if float(x["r"]) == r], key=lambda x: float(x["eta"]))
        axc.plot([float(x["eta"]) for x in sub], [float(x["rate_per_mode"]) for x in sub],
                 marker=markers[r], label=f"r={r}")
    axc.axvspan(0.28, 0.52, color="0.85", alpha=0.6)
    axc.set_xlabel(r"Transmissivity $\eta$")
    axc.set_ylabel(r"Concentration rate $c$ (nats/mode)")
    fig_style.legend_outside(axc)
    fig_style.panel_label(axc, "(a)")
    # panel (b): Wigner negativity vs eta (all r); threshold near eta=0.5
    axn = axes[1]
    for r in [0.3, 0.5, 1.0]:
        sub = sorted([x for x in rows if float(x["r"]) == r], key=lambda x: float(x["eta"]))
        axn.plot([float(x["eta"]) for x in sub], [float(x["negativity"]) for x in sub],
                 marker=markers[r], label=f"r={r}")
    axn.axvspan(0.28, 0.52, color="0.85", alpha=0.6)
    axn.axhline(0.0, color="0.5", lw=0.6)
    axn.set_xlabel(r"Transmissivity $\eta$")
    axn.set_ylabel("Wigner negativity $N$")
    fig_style.legend_outside(axn)
    fig_style.panel_label(axn, "(b)")
    fig_style.save(fig, "figures/fig5_concentration_vs_simulability")
    plt.close(fig)


def fig6_nongaussian_concentration():
    rows = _load("nongaussian_concentration.csv")
    fig, ax = fig_style.new_figure(kind="single", height_cm=7.0)
    series = [(0.5, 1.0, "o"), (0.5, 0.6, "s"), (0.5, 0.4, "^")]
    for r, eta, mk in series:
        pts = sorted([(int(x["M"]), float(x["var"])) for x in rows
                      if float(x["r"]) == r and float(x["eta"]) == eta])
        tag = "simulable" if eta < 0.55 else "hard"
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                    label=f"$\\eta$={eta} ({tag})")
    ax.set_xlabel("Number of modes $M$")
    ax.set_ylabel(r"$\mathrm{Var}_{x,x'}[k]$")
    fig_style.legend_outside(ax)
    fig_style.save(fig, "figures/fig6_nongaussian_concentration")
    plt.close(fig)


if __name__ == "__main__":
    fig1_concentration_vs_M(); print("fig1 done")
    fig2_rate_vs_loss(); print("fig2 done")
    fig3_bandwidth_compensation(); print("fig3 done")
    fig4_gbs_vs_gaussian(); print("fig4 done")
    fig5_concentration_vs_simulability(); print("fig5 done")
    fig6_nongaussian_concentration(); print("fig6 done")
