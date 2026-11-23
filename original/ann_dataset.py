"""
ANN training-dataset generator for the third-grade-fluid microchannel model.

Samples the 5 non-fixed inputs
    m   in {-2, 0, 0.5}      (reaction-kinetics index, discrete)
    Br  in [0, 0.5]          (Brinkman number)
    lam in [0, 1]            (Frank-Kamenetskii parameter)
    G   in [0, 2]            (pressure-gradient parameter)
    Gr  in [0, 3.5]          (buoyancy parameter)
solves the BVP (Eqs. 16-17 with BCs 18-21) for each sample, and records the
two ANN targets
    S_xy(y=1) = [du/dy + 2*Lambda*(du/dy)^3]_{y=1}     (Eq. 22)
    Nu(y=1)   = -dtheta/dy|_{y=1}                      (Eq. 23)

Samples whose BVP has no steady solution (thermal runaway) are discarded, so
the CSV covers only the physically valid region.

Fixed parameters: Lambda=0.5, Da=2, K=2, Omega_3=1, S_n=1, Omega=1, E_1=2,
                  gamma_1=0.01, lambda_1=0.01   (same as tgf_full_analysis.py)

Outputs (saved alongside this script):
    ann_training_dataset.csv   columns: m, Br, lambda, G, Gr, Sxy_wall, Nu_wall
    fig9_ann_dataset.png       targets vs each input, colored by m
"""

import os
import csv
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

warnings.filterwarnings("ignore")

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------- fixed parameters ----------------------------
Lam = 0.5
Da  = 2.0
K   = 2.0
Om3 = 1.0
S_n = 1.0
Om  = 1.0
E1  = 2.0
g1  = 0.01
l1  = 0.01

# ------------------------- sampling ranges -----------------------------
N_SAMPLES = 600          # attempts; non-converged samples are discarded
SEED      = 42
M_CHOICES = [-2.0, 0.0, 0.5]
BR_RANGE  = (0.0, 0.5)
LAM_RANGE = (0.0, 1.0)
G_RANGE   = (0.0, 2.0)
GR_RANGE  = (0.0, 3.5)


def phi(y):
    return np.cosh(K * y) / np.cosh(K)


def make_system(m, BrX, lamX, GX, GrX):
    def rhs(y, Y):
        u, p, th, q = Y
        arg = np.maximum(1.0 + Om * th, 1e-12)
        react = lamX * arg**m * np.exp(-E1 / arg)
        upp = -(GX - u / Da + GrX * th + K**2 * Om3 * phi(y)) / (1.0 + 6.0 * Lam * p**2)
        qpp = -(BrX * p**2 + 2.0 * Lam * BrX * p**4 + S_n + react)
        return np.vstack([p, upp, q, qpp])

    def bcs(Ya, Yb):
        return np.array([
            Ya[0] - g1 * (Ya[1] + 2.0 * Lam * Ya[1]**3),
            Yb[0] + g1 * (Yb[1] + 2.0 * Lam * Yb[1]**3) - 1.0,
            Ya[2] - l1 * Ya[3],
            Yb[2] + l1 * Yb[3] - 1.0,
        ])

    return rhs, bcs


def solve_sample(m, Br, lam, G, Gr):
    """Continuation: ramp (Br, lam, Gr) together from 0 to target.
    Returns (Sxy_wall, Nu_wall) or None if no steady solution."""
    y = np.linspace(-1.0, 1.0, 121)
    Y = np.zeros((4, y.size))
    Y[0] = 0.5 * (1.0 + y)
    Y[2] = 0.5 * (1.0 + y)
    for f in (0.25, 0.5, 0.75, 1.0):
        rhs, bcs = make_system(m, f * Br, f * lam, G, f * Gr)
        sol = solve_bvp(rhs, bcs, y, Y, tol=1e-6, max_nodes=20000)
        if not sol.success or not np.isfinite(sol.y).all() or abs(sol.y[2]).max() > 5e3:
            return None
        y, Y = sol.x, sol.y
    rhs, bcs = make_system(m, Br, lam, G, Gr)
    sol = solve_bvp(rhs, bcs, y, Y, tol=1e-8, max_nodes=40000)
    if not sol.success:
        return None
    p1, q1 = sol.sol(1.0)[1], sol.sol(1.0)[3]
    Sxy = p1 + 2.0 * Lam * p1**3
    Nu = -q1
    return Sxy, Nu


# --------------------------- generate dataset --------------------------
rng = np.random.default_rng(SEED)
rows, n_fail = [], 0
for i in range(N_SAMPLES):
    m   = M_CHOICES[rng.integers(len(M_CHOICES))]
    Br  = rng.uniform(*BR_RANGE)
    lam = rng.uniform(*LAM_RANGE)
    G   = rng.uniform(*G_RANGE)
    Gr  = rng.uniform(*GR_RANGE)
    res = solve_sample(m, Br, lam, G, Gr)
    if res is None:
        n_fail += 1
        continue
    rows.append([m, Br, lam, G, Gr, res[0], res[1]])
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_SAMPLES} attempted, {len(rows)} converged")

print(f"\nDataset: {len(rows)} converged samples "
      f"({n_fail} discarded, no steady solution)")

csv_path = os.path.join(OUTDIR, "ann_training_dataset.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "Br", "lambda", "G", "Gr", "Sxy_wall", "Nu_wall"])
    for r in rows:
        w.writerow([f"{r[0]:g}"] + [f"{v:.8e}" for v in r[1:]])
print(f"Saved {csv_path}")

# ------------------------------ plots ----------------------------------
data = np.array(rows)
colors = {-2.0: "#c0392b", 0.0: "#2471a3", 0.5: "#1e8449"}
labels = {-2.0: "m = -2", 0.0: "m = 0", 0.5: "m = 0.5"}
inputs = [("Br", 1), (r"$\lambda$", 2), ("G", 3), ("Gr", 4)]
targets = [(r"$S_{xy}(y=1)$", 5), ("Nu(y=1)", 6)]

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for row, (tlab, tcol) in enumerate(targets):
    for coli, (ilab, icol) in enumerate(inputs):
        ax = axes[row, coli]
        for m in M_CHOICES:
            sel = data[:, 0] == m
            ax.scatter(data[sel, icol], data[sel, tcol], s=10, alpha=0.6,
                       color=colors[m], label=labels[m] if (row == 0 and coli == 0) else None)
        ax.set_xlabel(ilab)
        if coli == 0:
            ax.set_ylabel(tlab)
fig.legend(loc="upper right", fontsize=10)
fig.suptitle(f"ANN training dataset: {len(rows)} BVP solutions  "
             r"(targets $S_{xy}(y=1)$, Nu$(y=1)$ vs the 4 continuous inputs, colored by m)",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005,
         f"fixed: $\\Lambda$={Lam}, Da={Da}, K={K}, $\\Omega_3$={Om3}, $S_n$={S_n}, "
         f"$\\Omega$={Om}, $E_1$={E1}, $\\gamma_1$={g1}, $\\lambda_1$={l1}",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, "fig9_ann_dataset.png"), dpi=160)
plt.close(fig)
print("Saved fig9_ann_dataset.png")
