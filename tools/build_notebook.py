"""Build notebooks/01_photonic_kernel_concentration.ipynb (mirrors src/, regenerates results+figures).
Run: python3 tools/build_notebook.py   (then execute with nbconvert)
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import pathlib

nb = new_notebook()
C = []

C.append(new_markdown_cell(
"# Loss-induced concentration of photonic quantum kernels\n"
"\n"
"Reproducibility notebook. It calls the project library in `src/` (the executable ground truth) "
"and regenerates every reported number and figure. Author: Hitesh Kumar Singh.\n"
"\n"
"Sections: (1) conventions & validation against qutip; (2) exact Gaussian concentration law; "
"(3) loss dependence of the rate; (4) bandwidth reversibility; (5) GBS feature kernel; (6) figures."))

C.append(new_code_cell(
"import sys, os, pathlib\n"
"# locate repo root (dir containing src/gaussian.py) and work from there\n"
"_p = pathlib.Path.cwd()\n"
"for _c in [_p, *_p.parents]:\n"
"    if (_c/'src'/'gaussian.py').exists():\n"
"        os.chdir(_c); sys.path.insert(0, str(_c)); break\n"
"import numpy as np\n"
"from src.gaussian import single_mode_kernel, effective_bandwidth, loss_covariance_diag\n"
"from src.concentration import exact_mean_var, exact_rate_per_mode, fit_decay_rate\n"
"np.set_printoptions(precision=6, suppress=True)\n"
"print('library imported')"))

C.append(new_markdown_cell(
"## 1. Conventions and validation\n"
"hbar=2 (vacuum covariance = I). The encoding shifts only the displacement, so both states share "
"one covariance and the squared-fidelity kernel is exactly `k = exp(-gamma dx^2)` per mode. We "
"validate against qutip Fock-space fidelity including a beamsplitter loss channel."))

C.append(new_code_cell(
"import qutip as qt\n"
"N=30\n"
"def k_qutip(rho1,rho2):\n"
"    return qt.fidelity(qt.ket2dm(rho1) if rho1.type=='ket' else rho1,\n"
"                       qt.ket2dm(rho2) if rho2.type=='ket' else rho2)**2\n"
"def lossy(psi, eta):\n"
"    th=np.arccos(np.sqrt(eta))\n"
"    a1=qt.tensor(qt.destroy(N),qt.qeye(N)); a2=qt.tensor(qt.qeye(N),qt.destroy(N))\n"
"    BS=(th*(a1.dag()*a2 - a1*a2.dag())).expm()\n"
"    out=BS*qt.tensor(psi, qt.basis(N,0)); return (out*out.dag()).ptrace(0)\n"
"v0=qt.basis(N,0); errs=[]\n"
"for r,eta,x,xp in [(0.0,1.0,0.3,-0.2),(0.6,0.5,0.8,0.2),(0.6,0.3,0.4,-0.3),(1.0,1.0,0.0,1.0)]:\n"
"    s=qt.squeeze(N,r)\n"
"    q1=qt.displace(N,x)*s*v0; q2=qt.displace(N,xp)*s*v0\n"
"    if eta<1.0: r1,r2=lossy(q1,eta),lossy(q2,eta)\n"
"    else: r1,r2=q1,q2\n"
"    k=single_mode_kernel(x-xp,r,eta); kq=k_qutip(r1,r2); errs.append(abs(k-kq))\n"
"    print(f'r={r} eta={eta} x={x:+.1f} x\\'={xp:+.1f}: k={k:.6f} qutip={kq:.6f} err={abs(k-kq):.2e}')\n"
"print('max |error| =', max(errs))\n"
"assert max(errs) < 1e-3"))

C.append(new_markdown_cell(
"## 2. Exact Gaussian concentration law\n"
"For iid Gaussian data, `Var[k] = (1+8 g sigma^2)^(-M/2) - (1+4 g sigma^2)^(-M)`, exponential in M "
"with per-mode rate `c = (1/2) ln(1+8 g sigma^2)`. We confirm the exponential fit quality."))

C.append(new_code_cell(
"sigma=1.0; disp=1.0; M_values=[1,2,4,8,16,32,50]\n"
"for r,eta in [(0.7,1.0),(0.7,0.5),(1.5,1.0)]:\n"
"    g=effective_bandwidth(r,eta,disp)\n"
"    V=[exact_mean_var(g,M,sigma)[1] for M in M_values]\n"
"    slope,inter,r2=fit_decay_rate(M_values,V)\n"
"    print(f'r={r} eta={eta}: gamma={g:.3f} rate/mode={exact_rate_per_mode(g,sigma):.4f} '\n"
"          f'fit_slope={-slope:.4f} R2={r2:.4f}')"))

C.append(new_markdown_cell(
"## 3. Loss dependence of the concentration rate\n"
"The effective bandwidth gamma increases with transmissivity eta (dgamma/deta = disp^2/v_x^2 > 0), "
"so the per-mode rate increases with eta: loss lowers the rate."))

C.append(new_code_cell(
"for r in [0.3,0.7,1.0]:\n"
"    line=[]\n"
"    for eta in [0.1,0.25,0.5,0.75,1.0]:\n"
"        g=effective_bandwidth(r,eta,1.0); line.append((eta,exact_rate_per_mode(g,1.0)))\n"
"    print(f'r={r}: '+'  '.join(f'eta={e:.2f}->c={c:.3f}' for e,c in line))"))

C.append(new_markdown_cell(
"## 4. Bandwidth reversibility\n"
"Because loss only rescales gamma, increasing the displacement restores the lossless gamma exactly."))

C.append(new_code_cell(
"r=0.7; g_target=effective_bandwidth(r,1.0,1.0)\n"
"for eta in [1.0,0.7,0.5,0.3,0.1]:\n"
"    v_x,_=loss_covariance_diag(r,eta); disp_needed=np.sqrt(g_target*v_x/eta)\n"
"    g_ach=effective_bandwidth(r,eta,disp_needed)\n"
"    print(f'eta={eta}: disp_needed={disp_needed:.4f} -> gamma={g_ach:.4f} (target {g_target:.4f})')"))

C.append(new_markdown_cell(
"## 5. GBS feature kernel (small-scale numerical exploration)\n"
"Photon-number-probability features via thewalrus, validated against qutip; cosine-similarity "
"kernel. Reported for small M as a finite-size observation."))

C.append(new_code_cell(
"from src.gbs import gbs_mean_var\n"
"rng=np.random.default_rng(20260615+99)\n"
"for r in [0.6]:\n"
"    for eta in [1.0,0.7,0.4]:\n"
"        for M,cut in [(1,10),(2,7),(3,5)]:\n"
"            mean,var,(lo,hi)=gbs_mean_var(r,M,eta,cut,0.8,80,2026,rng,1.0,n_boot=80)\n"
"            g=effective_bandwidth(r,eta,1.0); _,vg=exact_mean_var(g,M,0.8)\n"
"            print(f'r={r} eta={eta} M={M}: GBS Var={var:.3e} [{lo:.2e},{hi:.2e}]  Gaussian Var={vg:.3e}')"))

C.append(new_markdown_cell(
"## 6. Concentration vs classical-simulability boundary (non-Gaussian sector)\n"
"Encoding |chi(r)> = S(r)|1> (squeezed single photon), displaced by data, M-mode product, loss eta. "
"We compute the Wigner negativity (simulability resource: N>0 => classically hard) and the per-mode "
"concentration rate, showing they diverge: concentration persists where the kernel is classically "
"simulable."))

C.append(new_code_cell(
"from src.nongaussian import lossy_sigma, wigner_negativity, single_mode_moments, concentration_rate\n"
"print('  r    eta    negativity N   hard?   conc.rate c')\n"
"for r in [0.5]:\n"
"    for eta in [0.4, 0.6, 1.0]:\n"
"        sig=lossy_sigma(r,eta); N=wigner_negativity(sig)\n"
"        Ek,Ek2=single_mode_moments(sig,eta,0.8,n_grid=161); c=concentration_rate(Ek2)\n"
"        print(f'  {r}   {eta}     {N:.4f}        {int(N>1e-3)}      {c:.4f}')\n"
"print('-> eta=0.4 is classically simulable (N~0) yet still concentrates (c>0): boundaries diverge.')"))

C.append(new_markdown_cell(
"## 7. Regenerate the figure set\n"
"Calls the same figure script and style layer used for the manuscript figures."))

C.append(new_code_cell(
"import subprocess\n"
"print(subprocess.run([sys.executable,'src/make_figures.py'],capture_output=True,text=True).stdout)\n"
"print(subprocess.run([sys.executable,'tools/figure_qc.py','--project','.'],capture_output=True,text=True).stdout.splitlines()[-1])"))

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = pathlib.Path("notebooks/01_photonic_kernel_concentration.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)
