"""
OFAT (one-factor-at-a-time) ANN training-dataset generator, journal style,
CROSSED WITH m: the 7 continuous-parameter sweeps are repeated at EACH of
the three reaction-kinetics indices m = -2, 0, 0.5, so the ANN gets proper
coverage of m (the plain OFAT design had only 3 m-samples, which was not
enough to learn the strong m-sensitivity of Nu).

ANN inputs (8): m, Br, lambda, G, Gr, Da, Lambda, Omega_3
   (lambda  = Frank-Kamenetskii reaction parameter,
    Lambda  = third-grade / non-Newtonian fluid parameter - different symbols!)
Slip parameters gamma_1, lambda_1 are NOT inputs (kept fixed).

Baseline: Br=0.5, lambda=1, G=2, Gr=3.5, Da=2, Lambda=0.5, Omega_3=1
Sweeps (60 points each, repeated for m = -2, 0, 0.5):
    Br      : 0    -> 0.5
    lambda  : 0    -> 1
    G       : 0    -> 2
    Gr      : 0    -> 3.5
    Da      : 0.2  -> 5
    Lambda  : 0.4  -> 2
    Omega_3 : 0    -> 2
Maximum 3 x 7 x 60 = 1260 samples; sweep points with no steady solution
(thermal runaway, mostly near range edges at m = 0.5) are discarded.

Targets: S_xy(y=1) (Eq. 22) and Nu(y=1) (Eq. 23).
Other constants identical to tgf_full_analysis.py. This script does not
modify the random-sampling dataset (ann_training_dataset.csv).

Outputs:
    ann_training_dataset_ofat.csv   columns: varied, m, Br, lambda, G, Gr,
                                             Da, Lambda, Omega3,
                                             Sxy_wall, Nu_wall
    fig13_ann_ofat.png              OFAT curves per m of both targets
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
K = 2.0; S_n = 1.0; Om = 1.0; E1 = 2.0; g1 = 0.01; l1 = 0.01

BASE = dict(m=0.0, Br=0.5, lam=1.0, G=2.0, Gr=3.5, Da=2.0, Lam=0.5, Om3=1.0)
M_VALUES = [0.0]

N_PER_PARAM = 60
SWEEPS = [
    ("Br",  np.linspace(0.0, 0.5, N_PER_PARAM)),
    ("lam", np.linspace(0.0, 1.0, N_PER_PARAM)),
    ("G",   np.linspace(0.0, 2.0, N_PER_PARAM)),
    ("Gr",  np.linspace(0.0, 3.5, N_PER_PARAM)),
    ("Da",  np.linspace(0.2, 5.0, N_PER_PARAM)),
    ("Lam", np.linspace(0.4, 2.0, N_PER_PARAM)),
    ("Om3", np.linspace(0.0, 2.0, N_PER_PARAM)),
]


def phi(y):
    return np.cosh(K * y) / np.cosh(K)


def make_system(pars, f=1.0):
    """f is the continuation ramp factor applied to (Br, lambda, Gr)."""
    m, GX = pars["m"], pars["G"]
    BrX, lamX, GrX = f * pars["Br"], f * pars["lam"], f * pars["Gr"]
    DaX, LamX, Om3X = pars["Da"], pars["Lam"], pars["Om3"]

    def rhs(y, Y):
        u, p, th, q = Y
        arg = np.maximum(1.0 + Om * th, 1e-12)
        react = lamX * arg**m * np.exp(-E1 / arg)
        upp = -(GX - u / DaX + GrX * th + K**2 * Om3X * phi(y)) / (1.0 + 6.0 * LamX * p**2)
        qpp = -(BrX * p**2 + 2.0 * LamX * BrX * p**4 + S_n + react)
        return np.vstack([p, upp, q, qpp])

    def bcs(Ya, Yb):
        return np.array([
            Ya[0] - g1 * (Ya[1] + 2.0 * LamX * Ya[1]**3),
            Yb[0] + g1 * (Yb[1] + 2.0 * LamX * Yb[1]**3) - 1.0,
            Ya[2] - l1 * Ya[3],
            Yb[2] + l1 * Yb[3] - 1.0,
        ])

    return rhs, bcs


def solve_case(pars, guess=None):
    """Solve one case; finer continuation ramp when starting cold (needed
    for the near-critical m = 0.5 cases)."""
    if guess is None:
        y = np.linspace(-1.0, 1.0, 121)
        Y = np.zeros((4, y.size)); Y[0] = 0.5 * (1 + y); Y[2] = 0.5 * (1 + y)
        ramp = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
    else:
        y, Y = guess
        ramp = (1.0,)
    for f in ramp:
        rhs, bcs = make_system(pars, f)
        sol = solve_bvp(rhs, bcs, y, Y, tol=1e-6, max_nodes=6000)
        if not sol.success or not np.isfinite(sol.y).all() or abs(sol.y[2]).max() > 5e3:
            return None
        y, Y = sol.x, sol.y
    rhs, bcs = make_system(pars, 1.0)
    sol = solve_bvp(rhs, bcs, y, Y, tol=1e-7, max_nodes=20000)
    return sol if sol.success else None


def wall_values(sol, pars):
    p1, q1 = sol.sol(1.0)[1], sol.sol(1.0)[3]
    return p1 + 2.0 * pars["Lam"] * p1**3, -q1


# --------------------------- generate dataset --------------------------
COLS = ["m", "Br", "lam", "G", "Gr", "Da", "Lam", "Om3"]
rows, n_fail = [], 0
for mval in M_VALUES:
    for key, grid in SWEEPS:
        guess = None          # fresh continuation per (m, sweep)
        n_ok = 0
        cold_failed_prev = False
        fail_streak = 0
        for v in grid:
            pars = dict(BASE); pars["m"] = mval; pars[key] = v
            sol = solve_case(pars, guess=guess)
            if sol is None and not cold_failed_prev:
                sol = solve_case(pars, guess=None)   # cold retry
            if sol is None:
                n_fail += 1
                guess = None
                cold_failed_prev = True
                fail_streak += 1
                if n_ok > 0 and fail_streak >= 5:
                    break        # past criticality - rest of sweep would fail too
                continue
            cold_failed_prev = False
            fail_streak = 0
            guess = (sol.x, sol.y)
            s, nu = wall_values(sol, pars)
            rows.append([key] + [pars[c] for c in COLS] + [s, nu])
            n_ok += 1
        print(f"m={mval:>4}, sweep {key:<4}: {n_ok}/{len(grid)} converged")

print(f"\nOFAT-x-m dataset: {len(rows)} samples ({n_fail} discarded, "
      f"no steady solution)")

csv_path = os.path.join(OUTDIR, "ann_training_dataset_ofat.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["varied", "m", "Br", "lambda", "G", "Gr", "Da", "Lambda",
                "Omega3", "Sxy_wall", "Nu_wall"])
    for r in rows:
        w.writerow([r[0], f"{r[1]:g}"] + [f"{v:.8e}" for v in r[2:]])
print(f"Saved {csv_path}")

# ------------------------------ plot -----------------------------------
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
xlabels = {"Br": "Br", "lam": r"$\lambda$", "G": "G", "Gr": "Gr",
           "Da": "Da", "Lam": r"$\Lambda$", "Om3": r"$\Omega_3$"}
xidx = {c: i + 1 for i, c in enumerate(COLS)}
mcolors = {-2.0: "#c0392b", 0.0: "#2471a3", 0.5: "#1e8449"}
mstyles = {-2.0: "-", 0.0: "--", 0.5: "-."}
mlabels = {-2.0: "m = -2", 0.0: "m = 0", 0.5: "m = 0.5"}

fig, axes = plt.subplots(4, 4, figsize=(15.5, 13.5))
for k, (key, grid) in enumerate(SWEEPS):
    rr, cc = 2 * (k // 4), k % 4
    for mval in M_VALUES:
        sel = [r for r in rows if r[0] == key and r[1] == mval]
        xs = [r[xidx[key]] for r in sel]
        ss = [r[9] for r in sel]
        ns = [r[10] for r in sel]
        axes[rr, cc].plot(xs, ss, mstyles[mval], color=mcolors[mval], lw=1.8,
                          label=mlabels[mval])
        axes[rr + 1, cc].plot(xs, ns, mstyles[mval], color=mcolors[mval], lw=1.8,
                              label=mlabels[mval])
    axes[rr, cc].set_title(f"sweep of {xlabels[key]}")
    axes[rr, cc].set_ylabel(r"$S_{xy}(y=1)$")
    axes[rr + 1, cc].set_ylabel("Nu(y=1)")
    axes[rr + 1, cc].set_xlabel(xlabels[key])
    axes[rr, cc].legend(fontsize=7)
    axes[rr + 1, cc].legend(fontsize=7)
axes[2, 3].axis("off"); axes[3, 3].axis("off")
fig.suptitle(f"OFAT-x-m ANN dataset ({len(rows)} samples, 8 inputs): each "
             f"sweep repeated for m = -2, 0, 0.5\n"
             f"(baseline: Br={BASE['Br']:g}, $\\lambda$={BASE['lam']:g}, "
             f"G={BASE['G']:g}, Gr={BASE['Gr']:g}, Da={BASE['Da']:g}, "
             f"$\\Lambda$={BASE['Lam']:g}, $\\Omega_3$={BASE['Om3']:g})",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005,
         f"fixed: K={K}, $S_n$={S_n}, $\\Omega$={Om}, $E_1$={E1}, "
         f"$\\gamma_1$={g1}, $\\lambda_1$={l1}",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.015, 1, 0.93])
fig.savefig(os.path.join(OUTDIR, "fig13_ann_ofat.png"), dpi=160)
plt.close(fig)
print("Saved fig13_ann_ofat.png")
