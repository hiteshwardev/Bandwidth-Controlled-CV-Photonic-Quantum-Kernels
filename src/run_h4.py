"""H4: concentration-vs-simulability boundary for the non-Gaussian (squeezed single-photon) kernel.

For each (r, eta): Wigner negativity N (simulability resource; N>0 => classically hard via
multiplicative growth over modes) and the per-mode concentration rate c = -ln E[k^2] with the
product-kernel variance at several M. Also a large-M concentration table (exact via moments^M).

Run chunked by r: python3 -m src.run_h4 <r1,r2,...>
"""
import os, csv, sys, time
import numpy as np
from src.nongaussian import (lossy_sigma, wigner_negativity, single_mode_moments,
                             product_variance, concentration_rate)

OUT = "results/h4_boundary.csv"
OUTM = "results/nongaussian_concentration.csv"
FIELDS = ["r", "eta", "negativity", "Ek", "Ek2", "rate_per_mode",
          "var_M2", "var_M10", "var_M50", "classically_hard", "sigma_data"]
MFIELDS = ["r", "eta", "M", "var", "rate_per_mode"]


def run(r_vals, eta_vals=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), sigma_data=0.8):
    rows, mrows = [], []
    for r in r_vals:
        for eta in eta_vals:
            t0 = time.time()
            sigma = lossy_sigma(r, eta)
            N = wigner_negativity(sigma)
            Ek, Ek2 = single_mode_moments(sigma, eta, sigma_data)
            c = concentration_rate(Ek2)
            hard = N > 1e-3
            rows.append(dict(r=r, eta=eta, negativity=N, Ek=Ek, Ek2=Ek2, rate_per_mode=c,
                             var_M2=product_variance(Ek, Ek2, 2),
                             var_M10=product_variance(Ek, Ek2, 10),
                             var_M50=product_variance(Ek, Ek2, 50),
                             classically_hard=int(hard), sigma_data=sigma_data))
            for M in [1, 2, 4, 8, 16, 32, 50]:
                mrows.append(dict(r=r, eta=eta, M=M, var=product_variance(Ek, Ek2, M),
                                  rate_per_mode=c))
            print(f"  r={r} eta={eta}: N={N:.4f} hard={int(hard)} rate/mode={c:.4f} "
                  f"Var(M=10)={product_variance(Ek,Ek2,10):.3e} ({time.time()-t0:.1f}s)")
    _append(OUT, FIELDS, rows)
    _append(OUTM, MFIELDS, mrows)
    print(f"appended {len(rows)} boundary rows, {len(mrows)} concentration rows")


def _append(path, fields, rows):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


if __name__ == "__main__":
    r_vals = [float(v) for v in sys.argv[1].split(",")]
    run(r_vals)
