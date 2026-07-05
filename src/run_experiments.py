"""Run benchmark + production experiments; write results/*.csv and results/register.csv.

Usage: python3 src/run_experiments.py
Deterministic (seeds from src.config.SEED). Runtimes are CPU-seconds to a few minutes.
"""
from __future__ import annotations
import os, csv, json, time, hashlib
import numpy as np

from src.config import SEED, r_to_db
from src.gaussian import effective_bandwidth
from src.concentration import (exact_mean_var, exact_rate_per_mode, mc_mean_var,
                               fit_decay_rate, shots_to_resolve)
from src.gbs import gbs_mean_var, _haar_unitary
from src.gaussian import product_kernel

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

REGISTER = os.path.join(RESULTS, "register.csv")
_register_rows = []

def _runid(kind, params):
    h = hashlib.sha256((kind + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:8]
    return f"{kind}_{h}"

def _register(run_id, kind, params, outfile, status):
    _register_rows.append(dict(run_id=run_id, kind=kind, params=json.dumps(params, sort_keys=True),
                               output=outfile, status=status, seed=SEED,
                               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))

def write_register():
    with open(REGISTER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id","kind","params","output","status","seed","timestamp"])
        w.writeheader()
        for r in _register_rows:
            w.writerow(r)


# ----------------------------- production 1: Gaussian exact concentration -----------------------------

def run_gaussian_concentration():
    r_vals = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]
    eta_vals = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    M_vals = [1, 2, 4, 8, 16, 32, 50]
    sigma = 1.0; disp = 1.0
    out = os.path.join(RESULTS, "gaussian_concentration.csv")
    rows = []
    for r in r_vals:
        for eta in eta_vals:
            gamma = effective_bandwidth(r, eta, disp)
            variances = []
            for M in M_vals:
                _, var = exact_mean_var(gamma, M, sigma)
                variances.append(var)
                rows.append(dict(r=r, r_dB=round(r_to_db(r),3), eta=eta, M=M, gamma=gamma,
                                 sigma=sigma, var_exact=var,
                                 rate_per_mode=exact_rate_per_mode(gamma, sigma),
                                 shots_resolve=shots_to_resolve(var)))
            slope, intercept, r2 = fit_decay_rate(M_vals, variances)
            for row in rows[-len(M_vals):]:
                row["fit_slope"] = slope; row["fit_R2"] = r2
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for row in rows: w.writerow(row)
    _register(_runid("gauss_conc", {"r":r_vals,"eta":eta_vals,"M":M_vals}), "gaussian_concentration",
              {"r":r_vals,"eta":eta_vals,"M":M_vals,"sigma":sigma,"disp":disp}, out, "EXECUTED")
    return out


# ----------------------------- production 2: rate vs loss (and squeezing) -----------------------------

def run_rate_vs_loss():
    r_vals = [0.3, 0.7, 1.0]
    eta_grid = np.round(np.linspace(0.1, 1.0, 19), 3)
    sigma = 1.0; disp = 1.0
    out = os.path.join(RESULTS, "rate_vs_loss.csv")
    rows = []
    for r in r_vals:
        for eta in eta_grid:
            gamma = effective_bandwidth(r, eta, disp)
            rows.append(dict(r=r, eta=float(eta), gamma=gamma,
                             rate_per_mode=exact_rate_per_mode(gamma, sigma)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for row in rows: w.writerow(row)
    _register(_runid("rate_loss", {"r":r_vals}), "rate_vs_loss",
              {"r":r_vals,"eta":eta_grid.tolist(),"sigma":sigma,"disp":disp}, out, "EXECUTED")
    return out


# ----------------------------- production 3: bandwidth compensation (H3) -----------------------------

def run_bandwidth_compensation():
    """Show loss-induced bandwidth loss is exactly compensable by rescaling disp_scale (Gaussian sector)."""
    r = 0.7; sigma = 1.0
    gamma_target = effective_bandwidth(r, 1.0, 1.0)   # lossless baseline bandwidth
    eta_vals = [1.0, 0.7, 0.5, 0.3, 0.1]
    out = os.path.join(RESULTS, "bandwidth_compensation.csv")
    rows = []
    for eta in eta_vals:
        # gamma(eta, disp) = eta*disp^2/v_x ; solve disp so that gamma == gamma_target
        from src.gaussian import loss_covariance_diag
        v_x, _ = loss_covariance_diag(r, eta)
        disp_needed = np.sqrt(gamma_target * v_x / eta)
        gamma_check = effective_bandwidth(r, eta, disp_needed)
        # verify the M-concentration matches baseline at a few M
        match = []
        for M in [8, 16, 32]:
            _, v_base = exact_mean_var(gamma_target, M, sigma)
            _, v_comp = exact_mean_var(gamma_check, M, sigma)
            match.append(abs(v_base - v_comp))
        rows.append(dict(r=r, eta=eta, gamma_target=gamma_target, disp_needed=disp_needed,
                         gamma_achieved=gamma_check, max_var_mismatch=max(match)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for row in rows: w.writerow(row)
    _register(_runid("bw_comp", {"r":r}), "bandwidth_compensation",
              {"r":r,"eta":eta_vals,"gamma_target":gamma_target}, out, "EXECUTED")
    return out


# ----------------------------- production 4: GBS concentration (H2) -----------------------------

def run_gbs_concentration():
    rng = np.random.default_rng(SEED + 99)
    r_vals = [0.3, 0.6]
    eta_vals = [1.0, 0.7, 0.4]
    M_vals = [1, 2, 3, 4]
    cutoff_for_M = {1: 12, 2: 8, 3: 6, 4: 5}
    sigma = 0.8; disp = 1.0; n_pairs = 400; seed_U = 2026
    out = os.path.join(RESULTS, "gbs_concentration.csv")
    rows = []
    for r in r_vals:
        for eta in eta_vals:
            for M in M_vals:
                cutoff = cutoff_for_M[M]
                mean, var, (lo, hi) = gbs_mean_var(r, M, eta, cutoff, sigma, n_pairs,
                                                   seed_U, rng, disp_scale=disp, n_boot=150)
                # matched Gaussian product kernel variance (MC, same data scale) for comparison
                g = effective_bandwidth(r, eta, disp)
                _, var_gauss = exact_mean_var(g, M, sigma)
                rows.append(dict(r=r, eta=eta, M=M, cutoff=cutoff, sigma=sigma,
                                 gbs_mean=mean, gbs_var=var, gbs_var_lo=lo, gbs_var_hi=hi,
                                 gauss_var_exact=var_gauss, n_pairs=n_pairs))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for row in rows: w.writerow(row)
    _register(_runid("gbs_conc", {"r":r_vals,"eta":eta_vals,"M":M_vals}), "gbs_concentration",
              {"r":r_vals,"eta":eta_vals,"M":M_vals,"cutoff":cutoff_for_M,"sigma":sigma,
               "n_pairs":n_pairs,"seed_U":seed_U}, out, "EXECUTED")
    return out


if __name__ == "__main__":
    t0 = time.time()
    print("[1/4] Gaussian exact concentration grid ...");      print("   ->", run_gaussian_concentration())
    print("[2/4] Concentration rate vs loss ...");             print("   ->", run_rate_vs_loss())
    print("[3/4] Bandwidth compensation (H3) ...");            print("   ->", run_bandwidth_compensation())
    print("[4/4] GBS concentration (H2) ...");                 print("   ->", run_gbs_concentration())
    write_register()
    print(f"register -> {REGISTER}")
    print(f"done in {time.time()-t0:.1f} s")
