"""
Journal-style parametric-effects figures for the TGF microchannel model,
in the style of:
  - Journal 1 Figs. 3-4 / 11111.pdf Figs. 5-12:
      u(y) and theta(y) profiles, one parameter varied at a time
      (family of curves + trend arrow, other parameters fixed)
  - Journal 1 Fig. 5:
      wall quantities S_xy(y=1) and Nu(y=1) vs one parameter for
      several values of a second parameter

Baseline (fixed unless varied): m=0, Br=0.5, lambda=1, G=2, Gr=3.5,
Lambda=0.5, Da=2, K=2, Omega_3=1, S_n=1, Omega=1, E_1=2,
gamma_1=0.01, lambda_1=0.01   (same as tgf_full_analysis.py)

Outputs:
  fig10_velocity_params.png     u(y): effect of Br, lambda, G, Gr (2x2)
  fig11_temperature_params.png  theta(y): effect of Br, lambda, G, Gr (2x2)
  fig12_wall_params.png         S_xy and Nu vs parameter families (2x2)
"""

import os
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

warnings.filterwarnings("ignore")

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------- fixed parameters ----------------------------
Lam = 0.5; Da = 2.0; K = 2.0; Om3 = 1.0
S_n = 1.0; Om = 1.0; E1 = 2.0; g1 = 0.01; l1 = 0.01

# baseline values of the 5 study parameters
BASE = dict(m=0.0, Br=0.5, lam=1.0, G=2.0, Gr=3.5)


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


def solve_case(m, Br, lam, G, Gr, guess=None):
    """Continuation in (Br, lam, Gr); returns solve_bvp solution or None."""
    if guess is None:
        y = np.linspace(-1.0, 1.0, 121)
        Y = np.zeros((4, y.size)); Y[0] = 0.5 * (1 + y); Y[2] = 0.5 * (1 + y)
        ramp = (0.25, 0.5, 0.75, 1.0)
    else:
        y, Y = guess
        ramp = (1.0,)
    for f in ramp:
        rhs, bcs = make_system(m, f * Br, f * lam, G, f * Gr)
        sol = solve_bvp(rhs, bcs, y, Y, tol=1e-6, max_nodes=20000)
        if not sol.success or not np.isfinite(sol.y).all() or abs(sol.y[2]).max() > 5e3:
            return None
        y, Y = sol.x, sol.y
    rhs, bcs = make_system(m, Br, lam, G, Gr)
    sol = solve_bvp(rhs, bcs, y, Y, tol=1e-8, max_nodes=40000)
    return sol if sol.success else None


def wall_values(sol):
    p1, q1 = sol.sol(1.0)[1], sol.sol(1.0)[3]
    return p1 + 2.0 * Lam * p1**3, -q1     # S_xy(1), Nu(1)


plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120})
FIXNOTE = (f"fixed: m={BASE['m']:g}, $\\Lambda$={Lam}, Da={Da}, K={K}, "
           f"$\\Omega_3$={Om3}, $S_n$={S_n}, $\\Omega$={Om}, $E_1$={E1}, "
           f"$\\gamma_1$={g1}, $\\lambda_1$={l1}")

# parameter families for the profile figures
VARY = [
    ("Br", "Br",       [0.0, 0.125, 0.25, 0.375, 0.5]),
    ("lam", r"$\lambda$", [0.0, 0.5, 1.0, 1.5, 2.0]),   # wider gap; steady solution exists to ~2.25 at m=0
    ("G",  "G",        [0.0, 0.5, 1.0, 1.5, 2.0]),
    ("Gr", "Gr",       [0.0, 1.0, 2.0, 3.0, 3.5]),
]
yy = np.linspace(-1, 1, 601)
cmap = plt.cm.plasma
SHOW_ARROWS = False   # set True to draw the trend arrow (direction of increasing parameter)


def profile_figure(fname, comp, ylab, title):
    """comp: 0 for u, 2 for theta."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    for ax, (key, lab, vals) in zip(axes.flat, VARY):
        curves = []
        for i, v in enumerate(vals):
            pars = dict(BASE); pars[key] = v
            sol = solve_case(pars["m"], pars["Br"], pars["lam"], pars["G"], pars["Gr"])
            if sol is None:
                continue
            f = sol.sol(yy)[comp]
            curves.append(f)
            ax.plot(yy, f, lw=2, color=cmap(0.1 + 0.75 * i / (len(vals) - 1)),
                    label=f"{lab} = {v:g}")
        # trend arrow at the location of maximum spread (optional)
        if SHOW_ARROWS and len(curves) >= 2:
            spread = np.abs(curves[-1] - curves[0])
            j = int(np.argmax(spread))
            ax.annotate("", xy=(yy[j], curves[-1][j]), xytext=(yy[j], curves[0][j]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333"))
        ax.set_xlabel("y"); ax.set_ylabel(ylab)
        ax.set_title(f"effect of {lab}")
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005, FIXNOTE + "  (varied parameter's baseline replaced)",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(OUTDIR, fname), dpi=170); plt.close(fig)
    print("saved", fname)


profile_figure("fig10_velocity_params.png", 0, "u(y)",
               "Velocity profiles u(y) — effect of Br, $\\lambda$, G, Gr")
profile_figure("fig11_temperature_params.png", 2, r"$\theta$(y)",
               "Temperature profiles $\\theta$(y) — effect of Br, $\\lambda$, G, Gr")

# ------------- wall-quantity families (Journal 1 Fig. 5 style) ---------
def sweep(key_x, xs, key_c, cval):
    """Sweep wall values along key_x grid with warm starts; key_c fixed at cval."""
    pars = dict(BASE); pars[key_c] = cval; pars[key_x] = xs[0]
    sol = solve_case(pars["m"], pars["Br"], pars["lam"], pars["G"], pars["Gr"])
    if sol is None:
        return [], [], []
    out_x, out_s, out_n = [], [], []
    guess = (sol.x, sol.y)
    for x in xs:
        pars[key_x] = x
        sol = solve_case(pars["m"], pars["Br"], pars["lam"], pars["G"], pars["Gr"],
                         guess=guess)
        if sol is None:
            break
        guess = (sol.x, sol.y)
        s, n = wall_values(sol)
        out_x.append(x); out_s.append(s); out_n.append(n)
    return out_x, out_s, out_n


fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
markers = ["-o", "--s", "-.^", ":d"]

# (a) S_xy vs Br for several Gr     (b) S_xy vs G for several Gr
# (c) Nu vs Br for several lambda   (d) Nu vs lambda for several Br
Br_grid = np.linspace(0, 0.5, 11)
G_grid = np.linspace(0, 2, 11)
lam_grid = np.linspace(0, 1, 11)

for i, Grv in enumerate([1.0, 2.0, 3.0, 3.5]):
    x, s, _ = sweep("Br", Br_grid, "Gr", Grv)
    axes[0, 0].plot(x, s, markers[i], ms=4, lw=1.8, label=f"Gr = {Grv:g}")
    x, s, _ = sweep("G", G_grid, "Gr", Grv)
    axes[0, 1].plot(x, s, markers[i], ms=4, lw=1.8, label=f"Gr = {Grv:g}")
for i, lv in enumerate([0.25, 0.5, 0.75, 1.0]):
    x, _, n = sweep("Br", Br_grid, "lam", lv)
    axes[1, 0].plot(x, n, markers[i], ms=4, lw=1.8, label=f"$\\lambda$ = {lv:g}")
for i, bv in enumerate([0.125, 0.25, 0.375, 0.5]):
    x, _, n = sweep("lam", lam_grid, "Br", bv)
    axes[1, 1].plot(x, n, markers[i], ms=4, lw=1.8, label=f"Br = {bv:g}")

axes[0, 0].set_xlabel("Br"); axes[0, 0].set_ylabel(r"$S_{xy}(y=1)$")
axes[0, 0].set_title(r"(a) $S_{xy}$ vs Br for several Gr")
axes[0, 1].set_xlabel("G"); axes[0, 1].set_ylabel(r"$S_{xy}(y=1)$")
axes[0, 1].set_title(r"(b) $S_{xy}$ vs G for several Gr")
axes[1, 0].set_xlabel("Br"); axes[1, 0].set_ylabel("Nu(y=1)")
axes[1, 0].set_title(r"(c) Nu vs Br for several $\lambda$")
axes[1, 1].set_xlabel(r"$\lambda$"); axes[1, 1].set_ylabel("Nu(y=1)")
axes[1, 1].set_title(r"(d) Nu vs $\lambda$ for several Br")
for a in axes.flat:
    a.legend(fontsize=9)
fig.suptitle("Wall shear stress (Eq. 22) and Nusselt number (Eq. 23) — "
             "parameter families", fontsize=14, fontweight="bold")
fig.text(0.5, 0.005, FIXNOTE, ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(os.path.join(OUTDIR, "fig12_wall_params.png"), dpi=170)
plt.close(fig)
print("saved fig12_wall_params.png")
