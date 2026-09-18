"""
==============================================================================
Q4_LASSO_ADMM.py
==============================================================================
AI3403: Multi-Agent Systems -- Assignment 3, Problem 4
IIT Hyderabad

WHAT THIS FILE DOES
--------------------
Solves the LASSO problem
        minimize_x  (1/2)||A x - b||_2^2 + lambda ||x||_1
via ADMM, using the standard variable-splitting x = z, with closed-form
x- and z-updates (ridge-regression solve + soft-thresholding).

1. Generates a synthetic sparse-regression problem (true sparse signal
   x_true, random design matrix A, noisy observations b = A x_true + noise).
2. Runs ADMM to solve the LASSO problem for x.
3. Plots: objective-value convergence, primal & dual residuals, and the
   recovered sparse signal against the true signal.

HOW TO RUN
----------
    python Q4_LASSO_ADMM.py

OUTPUTS (written to ./figures/)
--------------------------------
    figures/q4_objective_convergence.png
    figures/q4_residuals.png
    figures/q4_sparse_signal_recovery.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# 0. Reproducibility and output folder
# -----------------------------------------------------------------------
SEED = 11
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 1. Synthetic sparse-regression data
# -----------------------------------------------------------------------
n_features = 100     # length of x
n_samples = 40        # number of rows of A (under-determined: n_samples < n_features)
sparsity = 8           # number of non-zero entries in the true signal
noise_std = 0.05

x_true = np.zeros(n_features)
support = rng.choice(n_features, size=sparsity, replace=False)
x_true[support] = rng.uniform(-3, 3, size=sparsity)

A = rng.normal(0, 1.0 / np.sqrt(n_samples), size=(n_samples, n_features))
noise = rng.normal(0, noise_std, size=n_samples)
b = A @ x_true + noise

LAMBDA = 0.15   # L1 regularisation weight
print(f"LASSO problem: n_features={n_features}, n_samples={n_samples}, "
      f"true sparsity={sparsity}, lambda={LAMBDA}")

# -----------------------------------------------------------------------
# 2. ADMM derivation (see report for the full step-by-step derivation)
#
#    Split:      minimize_{x,z} (1/2)||Ax-b||^2 + lambda||z||_1   s.t. x = z
#    Augmented Lagrangian (scaled dual u):
#       L_rho(x,z,u) = (1/2)||Ax-b||^2 + lambda||z||_1
#                      + (rho/2)||x - z + u||^2 - (rho/2)||u||^2
#
#    x-update (unconstrained quadratic -> closed form):
#       x^{k+1} = (A^T A + rho I)^{-1} ( A^T b + rho (z^k - u^k) )
#
#    z-update (soft-thresholding / proximal operator of lambda*||.||_1):
#       z^{k+1} = S_{lambda/rho} ( x^{k+1} + u^k ),
#       S_kappa(a) = sign(a) * max(|a| - kappa, 0)     (component-wise)
#
#    Dual update:
#       u^{k+1} = u^k + x^{k+1} - z^{k+1}
# -----------------------------------------------------------------------
def soft_threshold(a, kappa):
    return np.sign(a) * np.maximum(np.abs(a) - kappa, 0.0)

def run_admm_lasso(A, b, lam, rho=1.0, n_iter=300):
    n_feat = A.shape[1]
    x = np.zeros(n_feat)
    z = np.zeros(n_feat)
    u = np.zeros(n_feat)

    AtA = A.T @ A
    Atb = A.T @ b
    # Pre-factorise (A^T A + rho I) once -- it does not change across iterations
    lhs = AtA + rho * np.eye(n_feat)
    lhs_inv = np.linalg.inv(lhs)

    objective_hist, primal_res_hist, dual_res_hist = [], [], []
    for k in range(n_iter):
        x = lhs_inv @ (Atb + rho * (z - u))

        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)

        u = u + x - z

        objective_hist.append(0.5 * np.sum((A @ x - b) ** 2) + lam * np.sum(np.abs(x)))
        primal_res_hist.append(np.linalg.norm(x - z))
        dual_res_hist.append(rho * np.linalg.norm(z - z_old))

    return z, objective_hist, primal_res_hist, dual_res_hist

x_hat, obj_hist, primal_res, dual_res = run_admm_lasso(A, b, LAMBDA)

recovered_support = np.where(np.abs(x_hat) > 1e-3)[0]
print(f"Recovered support size: {len(recovered_support)} (true sparsity: {sparsity})")
print(f"Final objective value: {obj_hist[-1]:.6f}")
print(f"||x_hat - x_true||_2 = {np.linalg.norm(x_hat - x_true):.4f}")
print(f"Final primal residual ||x-z||: {primal_res[-1]:.3e}, "
      f"dual residual: {dual_res[-1]:.3e}")

# -----------------------------------------------------------------------
# 3. Plot: objective-value convergence
# -----------------------------------------------------------------------
plt.figure(figsize=(7.5, 4.5))
plt.plot(obj_hist, color="tab:blue")
plt.xlabel("ADMM iteration $k$")
plt.ylabel(r"$\frac{1}{2}\|Ax-b\|_2^2 + \lambda\|x\|_1$")
plt.title("Q4: LASSO Objective-Value Convergence (ADMM)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q4_objective_convergence.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 4. Plot: primal and dual residuals
# -----------------------------------------------------------------------
plt.figure(figsize=(7.5, 4.5))
plt.plot(primal_res, label=r"Primal residual $\|x-z\|_2$")
plt.plot(dual_res, label=r"Dual residual $\rho\|z^k - z^{k-1}\|_2$")
plt.yscale("log")
plt.xlabel("ADMM iteration $k$")
plt.ylabel("Residual norm (log scale)")
plt.title("Q4: ADMM Primal and Dual Residuals")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q4_residuals.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 5. Plot: recovered sparse signal vs. true signal
# -----------------------------------------------------------------------
plt.figure(figsize=(9, 4.5))
markerline, stemlines, baseline = plt.stem(
    np.arange(n_features), x_true, linefmt="tab:red", markerfmt="ro", basefmt=" "
)
plt.setp(stemlines, alpha=0.5)
plt.setp(markerline, markersize=5, label="True signal $x_{true}$")

markerline2, stemlines2, baseline2 = plt.stem(
    np.arange(n_features), x_hat, linefmt="tab:blue", markerfmt="bx", basefmt=" "
)
plt.setp(stemlines2, alpha=0.5)
plt.setp(markerline2, markersize=6, label=r"ADMM-LASSO recovery $\hat{x}$")

plt.xlabel("Feature index")
plt.ylabel("Coefficient value")
plt.title(f"Q4: Sparse-Signal Recovery ($n$={n_features}, $m$={n_samples}, "
          f"true sparsity={sparsity})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q4_sparse_signal_recovery.png"), dpi=150)
plt.close()

print("\nQ4 complete. All figures saved under ./figures/")
