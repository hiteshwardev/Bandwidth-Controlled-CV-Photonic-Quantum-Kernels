"""Configuration and physical conventions for the photonic-kernel concentration study.

Conventions (fixed, project-wide):
  * hbar = 2, so the vacuum covariance matrix is the identity I_{2n}.
  * Quadrature ordering per mode is (x, p) with x = a + a^dagger, p = -i(a - a^dagger),
    giving [x, p] = 2i and vacuum variance Var(x) = Var(p) = 1.
  * Multimode ordering is mode-blocked: (x1, p1, x2, p2, ...).
  * Symmetric covariance V_ij = (1/2)<{Delta R_i, Delta R_j}>, so vacuum -> I.

All randomness is seeded from SEED for reproducibility.
"""
from __future__ import annotations
import numpy as np

SEED = 20260615

# Squeezing convention: squeezed vacuum covariance per mode = diag(e^{-2r}, e^{+2r}).
# Encoding: data x in R is mapped to a real displacement of magnitude (DISP_SCALE * x)
# applied to a fixed squeezed vacuum with squeezing R_SQUEEZE (nepers).
DISP_SCALE_DEFAULT = 1.0
R_SQUEEZE_DEFAULT = 0.7  # ~6.1 dB

# Parameter ranges for sensitivity / production (mirrors PREREGISTRATION.md section 6).
R_RANGE = (0.0, 1.5)            # squeezing in nepers (~0 to ~13 dB)
M_RANGE = (1, 50)              # number of modes
ETA_RANGE = (0.1, 1.0)        # per-mode loss transmissivity (1.0 = lossless)
SIGMA_DATA_RANGE = (0.5, 2.0)  # std of the data distribution

# dB conversion for squeezing variance reduction e^{-2r}: dB = (20/ln10) * r.
def r_to_db(r: float) -> float:
    return (20.0 / np.log(10.0)) * r

def symplectic_form(n_modes: int) -> np.ndarray:
    """Block-diagonal symplectic form Omega with per-mode [[0,1],[-1,0]] (x,p ordering)."""
    w = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return np.kron(np.eye(n_modes), w)
