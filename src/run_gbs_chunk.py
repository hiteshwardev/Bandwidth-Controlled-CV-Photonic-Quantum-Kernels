"""Chunked GBS concentration runner: append rows for a given (r-subset, M-subset).
Lets the expensive M=3 cases be split across calls to respect execution limits.
"""
import os, csv, sys, time
import numpy as np
from src.config import SEED
from src.gbs import gbs_mean_var
from src.gaussian import effective_bandwidth
from src.concentration import exact_mean_var

OUT = "results/gbs_concentration.csv"
FIELDS = ["r","eta","M","cutoff","sigma","gbs_mean","gbs_var","gbs_var_lo","gbs_var_hi",
          "gauss_var_exact","n_pairs"]
CUTOFF = {1: 10, 2: 7, 3: 5}

def run(r_vals, M_vals, eta_vals=(1.0, 0.7, 0.4), n_pairs=100, sigma=0.8, disp=1.0, seed_U=2026):
    rng = np.random.default_rng(SEED + 99)
    rows = []
    for r in r_vals:
        for eta in eta_vals:
            for M in M_vals:
                cut = CUTOFF[M]
                t0 = time.time()
                mean, var, (lo, hi) = gbs_mean_var(r, M, eta, cut, sigma, n_pairs, seed_U, rng,
                                                   disp_scale=disp, n_boot=120)
                g = effective_bandwidth(r, eta, disp)
                _, vg = exact_mean_var(g, M, sigma)
                rows.append(dict(r=r, eta=eta, M=M, cutoff=cut, sigma=sigma, gbs_mean=mean,
                                 gbs_var=var, gbs_var_lo=lo, gbs_var_hi=hi, gauss_var_exact=vg,
                                 n_pairs=n_pairs))
                print(f"  r={r} eta={eta} M={M}: gbs_var={var:.3e} gauss_var={vg:.3e} ({time.time()-t0:.1f}s)")
    exists = os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"appended {len(rows)} rows to {OUT}")

if __name__ == "__main__":
    # args: r_subset (csv) M_subset (csv); e.g. "0.3,0.6" "1,2"
    r_vals = [float(v) for v in sys.argv[1].split(",")]
    M_vals = [int(v) for v in sys.argv[2].split(",")]
    run(r_vals, M_vals)
