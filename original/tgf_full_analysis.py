"""
Combined analysis script: merges solve_tgf_bvp.py and nusselt_plot.py.

Part 1 (from solve_tgf_bvp.py):
  Solves the dimensionless momentum (Eq. 16) and energy (Eq. 17) equations
  for electroosmotic flow of a third-grade fluid in a vertical microchannel,
  with velocity-slip (Eqs. 18-19) and thermal-slip (Eqs. 20-21) BCs,
  for m = -2, 0, 0.5.  Produces fig1-fig5.

Part 2 (from nusselt_plot.py):
  Nusselt-number study Nu(y=1) = -dtheta/dy|_{y=1} (Eq. 23):
  (a) Nu vs Brinkman number Br, (b) Nu vs Frank-Kamenetskii lambda.
  Produces fig8_nusselt.png.

Outputs saved alongside this script:
  Figures : fig1_velocity.png, fig2_temperature.png,
            fig3_shear.png  (wall shear S_xy(y=1), Eq. 22, vs Br and vs G),
            fig5_combined.png (all results in one 3x2 diagram),
            fig8_nusselt.png
  CSV     : results_profiles.csv        (y, u, u', theta, theta' per m)
            results_summary.csv         (u_max, theta_max, S_xy, Nu per m)
            results_nusselt_vs_Br.csv   (Nu vs Br per m)
            results_nusselt_vs_lambda.csv (Nu vs lambda per m)
  NumPy   : results_profiles.npy, results_nusselt_vs_Br.npy,
            results_nusselt_vs_lambda.npy   (dict archives; load with
            np.load(f, allow_pickle=True).item())
            solution_data.npz (same format as original script)
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


def out(name):
    return os.path.join(OUTDIR, name)


# ----------------------------- parameters ------------------------------
Lam  = 0.5    # Lambda, non-Newtonian (third-grade) parameter
G    = 2.0    # pressure-gradient parameter
Da   = 2.0    # Darcy number (porous medium)
Gr   = 3.5    # buoyancy (Grashof) parameter (3.0 -> 3.5: stronger buoyancy so m affects u more)
K    = 2.0    # electroosmotic parameter
Om3  = 1.0    # Omega_3, electroosmotic velocity ratio
Br   = 0.5    # Brinkman number (requested 2 -> no steady solution)
S_n  = 1.0    # Joule-heating parameter (2.0 -> 1.0: less m-independent heating)
lam0 = 1.0    # Frank-Kamenetskii parameter (baseline)
Om   = 1.0    # Omega, temperature-difference ratio (0.5 -> 1.0: reaction term more sensitive to m)
E1   = 2.0    # activation-energy parameter
g1   = 0.01   # gamma_1, velocity-slip parameter
l1   = 0.01   # lambda_1, thermal-slip parameter

M_VALUES = [-2.0, 0.0, 0.5]
M_LABELS = {-2.0: "m = -2 (sensitized)",
             0.0: "m = 0 (Arrhenius)",
             0.5: "m = 0.5 (bimolecular)"}
colors = {-2.0: "#c0392b", 0.0: "#2471a3", 0.5: "#1e8449"}
styles = {-2.0: "-", 0.0: "--", 0.5: "-."}
mstyles = {-2.0: "-o", 0.0: "--s", 0.5: "-.^"}


def phi(y):
    return np.cosh(K * y) / np.cosh(K)


# --------------------------- ODE system --------------------------------
def make_system(m, BrX, lamX, GX=None):
    GX = G if GX is None else GX

    def rhs(y, Y):
        u, p, th, q = Y
        arg = np.maximum(1.0 + Om * th, 1e-12)
        react = lamX * arg**m * np.exp(-E1 / arg)
        upp = -(GX - u / Da + Gr * th + K**2 * Om3 * phi(y)) / (1.0 + 6.0 * Lam * p**2)
        qpp = -(BrX * p**2 + 2.0 * Lam * BrX * p**4 + S_n + react)
        return np.vstack([p, upp, q, qpp])

    def bcs(Ya, Yb):
        return np.array([
            Ya[0] - g1 * (Ya[1] + 2.0 * Lam * Ya[1]**3),        # Eq. 18
            Yb[0] + g1 * (Yb[1] + 2.0 * Lam * Yb[1]**3) - 1.0,  # Eq. 19
            Ya[2] - l1 * Ya[3],                                 # Eq. 20
            Yb[2] + l1 * Yb[3] - 1.0,                           # Eq. 21
        ])

    return rhs, bcs


# ======================================================================
# PART 1 : profile solutions (from solve_tgf_bvp.py)
# ======================================================================
def solve_for_m(m):
    """Continuation in Br from 0 up to the target value, then a tight solve."""
    y = np.linspace(-1.0, 1.0, 201)
    Y = np.zeros((4, y.size))
    Y[0] = 0.5 * (1.0 + y)
    Y[2] = 0.5 * (1.0 + y)
    for BrX in [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, Br]:
        rhs, bcs = make_system(m, BrX, lam0)
        sol = solve_bvp(rhs, bcs, y, Y, tol=1e-6, max_nodes=60000)
        if not sol.success:
            raise RuntimeError(f"continuation failed at Br={BrX}, m={m}: {sol.message}")
        y, Y = sol.x, sol.y
    rhs, bcs = make_system(m, Br, lam0)
    sol = solve_bvp(rhs, bcs, y, Y, tol=1e-9, max_nodes=120000)
    if not sol.success:
        raise RuntimeError(f"final solve failed for m={m}: {sol.message}")
    return sol


solutions = {m: solve_for_m(m) for m in M_VALUES}

# ------------------------ engineering quantities -----------------------
print(f"{'case':<26}{'u_max':>10}{'theta_max':>12}{'S_xy(y=1)':>12}{'Nu(y=1)':>10}")
results = {}
for m, sol in solutions.items():
    yy = np.linspace(-1, 1, 2001)
    u, p, th, q = sol.sol(yy)
    Sxy = p[-1] + 2.0 * Lam * p[-1]**3        # Eq. 22 at y = +1
    Nu  = -q[-1]                              # Eq. 23 at y = +1
    results[m] = dict(y=yy, u=u, up=p, th=th, q=q, Sxy=Sxy, Nu=Nu)
    print(f"{M_LABELS[m]:<26}{u.max():>10.5f}{th.max():>12.5f}{Sxy:>12.5f}{Nu:>10.5f}")

# ------------------------------- plots ---------------------------------
plt.rcParams.update({"font.size": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 120})
PSTR = (f"$\\Lambda$={Lam}, G={G}, Da={Da}, Gr={Gr}, K={K}, $\\Omega_3$={Om3}, "
        f"Br={Br}, $S_n$={S_n}, $\\lambda$={lam0}, $\\Omega$={Om}, $E_1$={E1}, "
        f"$\\gamma_1$={g1}, $\\lambda_1$={l1}")


def mplot(fname, key, ylab, title, transform=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for m in M_VALUES:
        r = results[m]
        v = r[key] if transform is None else transform(r)
        ax.plot(r["y"], v, styles[m], color=colors[m], lw=2.2, label=M_LABELS[m])
    ax.set_xlabel("y"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend()
    fig.text(0.5, 0.005, PSTR, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out(fname), dpi=170); plt.close(fig)


mplot("fig1_velocity.png", "u", "u(y)",
      "Velocity profiles - effect of reaction-kinetics index m")
mplot("fig2_temperature.png", "th", r"$\theta$(y)",
      "Temperature profiles - effect of reaction-kinetics index m")

# --------------------------- verification ------------------------------
print("\nVerification (max ODE residuals and BC errors):")
for m, sol in solutions.items():
    yv = np.linspace(-1, 1, 4001)
    u, p, th, q = sol.sol(yv)
    dY = sol.sol(yv, 1)
    upp, qpp = dY[1], dY[3]
    arg = 1.0 + Om * th
    res16 = upp + 6*Lam*p**2*upp + G - u/Da + Gr*th + K**2*Om3*phi(yv)
    res17 = qpp + Br*p**2 + 2*Lam*Br*p**4 + S_n + lam0*arg**m*np.exp(-E1/arg)
    _, bcs = make_system(m, Br, lam0)
    bc = bcs(sol.sol(-1.0), sol.sol(1.0))
    print(f"  m={m:>4}:  |res16|max={np.abs(res16).max():.2e}  "
          f"|res17|max={np.abs(res17).max():.2e}  |BC|max={np.abs(bc).max():.2e}")


# ======================================================================
# PART 2 : Nusselt-number study (from nusselt_plot.py)
# ======================================================================
def attempt(m, BrX, lamX, y, Y, GX=None):
    rhs, bc = make_system(m, BrX, lamX, GX)
    sol = solve_bvp(rhs, bc, y, Y, tol=1e-8, max_nodes=60000)
    ok = sol.success and np.isfinite(sol.y).all() and abs(sol.y[2]).max() < 5e3
    return ok, sol


def wall_shear(sol):
    """S_xy = (u' + 2*Lam*u'^3) at y = +1   (Eq. 22)."""
    p1 = sol.sol(1.0)[1]
    return p1 + 2.0 * Lam * p1**3


nusselt_Br = {}       # m -> (Br array, Nu array)
nusselt_lam = {}      # m -> (lambda array, Nu array)
shear_Br = {}         # m -> (Br array, S_xy(y=1) array)
shear_G = {}          # m -> (G array, S_xy(y=1) array)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

# ---- (a) Nu vs Br (lambda = 1) ----
Br_grid = np.arange(0.0, 0.501, 0.05)   # 0.6 -> 0.5: no steady solution beyond ~0.5 with new params
for m in M_VALUES:
    y = np.linspace(-1, 1, 151); Y = np.zeros((4, y.size))
    Y[0] = 0.5*(1+y); Y[2] = 0.5*(1+y)
    Brs, Nus, Sxys = [], [], []
    for BrX in Br_grid:
        ok, sol = attempt(m, BrX, 1.0, y, Y)
        if not ok:
            break
        y, Y = sol.x, sol.y
        Brs.append(BrX); Nus.append(-sol.sol(1.0)[3]); Sxys.append(wall_shear(sol))
    nusselt_Br[m] = (np.array(Brs), np.array(Nus))
    shear_Br[m] = (np.array(Brs), np.array(Sxys))
    axes[0].plot(Brs, Nus, mstyles[m], color=colors[m], lw=2, ms=5, label=M_LABELS[m])
axes[0].set_xlabel("Brinkman number  Br")
axes[0].set_ylabel("Nu(y=1)")
axes[0].set_title(r"(a) Nusselt number vs Br   ($\lambda$=1)")

# ---- (b) Nu vs lambda (Br = 0.5) ----
lam_grid = np.arange(0.0, 1.001, 0.1)   # 2.0 -> 1.0: no steady solution beyond ~1 for m=0.5 with new params
for m in M_VALUES:
    y = np.linspace(-1, 1, 151); Y = np.zeros((4, y.size))
    Y[0] = 0.5*(1+y); Y[2] = 0.5*(1+y)
    # ramp Br to 0.5 first at lambda=0
    for BrX in [0.0, 0.25, 0.5]:
        ok, sol = attempt(m, BrX, 0.0, y, Y); y, Y = sol.x, sol.y
    lams, Nus = [], []
    for lamX in lam_grid:
        ok, sol = attempt(m, 0.5, lamX, y, Y)
        if not ok:
            break
        y, Y = sol.x, sol.y
        lams.append(lamX); Nus.append(-sol.sol(1.0)[3])
    nusselt_lam[m] = (np.array(lams), np.array(Nus))
    axes[1].plot(lams, Nus, mstyles[m], color=colors[m], lw=2, ms=5, label=M_LABELS[m])
axes[1].set_xlabel(r"Frank-Kamenetskii parameter  $\lambda$")
axes[1].set_ylabel("Nu(y=1)")
axes[1].set_title("(b) Nusselt number vs $\\lambda$   (Br=0.5)")

for a in axes:
    a.grid(alpha=0.3); a.legend(fontsize=10)
fig.suptitle("Wall heat transfer, Eq. (23):  Nu = $-d\\theta/dy|_{y=1}$",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005, f"G={G}, Da={Da}, Gr={Gr}, K={K}, $\\Omega_3$={Om3}, $S_n$={S_n}, "
         f"$\\Omega$={Om}, $E_1$={E1}, $\\Lambda$={Lam}, $\\gamma_1$={g1}, $\\lambda_1$={l1}",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig(out("fig8_nusselt.png"), dpi=170); plt.close(fig)


# ======================================================================
# PART 2b : wall shear stress S_xy(y=1), Eq. (22)
#           S_xy is evaluated AT the wall (y=1), so it is a single value
#           per case -> plotted against Br and G, not against y.
# ======================================================================
# ---- S_xy(1) vs G  (Br = 0.5, lambda = 1) ----
G_grid = np.arange(0.0, 2.001, 0.2)   # 4.0 -> 2.0: m=0.5 has no steady solution beyond G~2.3 with new params
for m in M_VALUES:
    y = np.linspace(-1, 1, 151); Y = np.zeros((4, y.size))
    Y[0] = 0.5*(1+y); Y[2] = 0.5*(1+y)
    # ramp Br to 0.5 first at G = 0
    for BrX in [0.0, 0.25, 0.5]:
        ok, sol = attempt(m, BrX, lam0, y, Y, GX=0.0); y, Y = sol.x, sol.y
    Gs, Sxys = [], []
    for GX in G_grid:
        ok, sol = attempt(m, 0.5, lam0, y, Y, GX=GX)
        if not ok:
            break
        y, Y = sol.x, sol.y
        Gs.append(GX); Sxys.append(wall_shear(sol))
    shear_G[m] = (np.array(Gs), np.array(Sxys))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for m in M_VALUES:
    Brs, Sxys = shear_Br[m]
    axes[0].plot(Brs, Sxys, mstyles[m], color=colors[m], lw=2, ms=5, label=M_LABELS[m])
    Gs, Sxys = shear_G[m]
    axes[1].plot(Gs, Sxys, mstyles[m], color=colors[m], lw=2, ms=5, label=M_LABELS[m])
axes[0].set_xlabel("Brinkman number  Br")
axes[0].set_ylabel(r"$S_{xy}(y=1)$")
axes[0].set_title(r"(a) wall shear stress vs Br   (G=2, $\lambda$=1)")
axes[1].set_xlabel("pressure-gradient parameter  G")
axes[1].set_ylabel(r"$S_{xy}(y=1)$")
axes[1].set_title(r"(b) wall shear stress vs G   (Br=0.5, $\lambda$=1)")
for a in axes:
    a.grid(alpha=0.3); a.legend(fontsize=10)
fig.suptitle("Wall shear stress, Eq. (22):  "
             r"$S_{xy}=\left[\frac{du}{dy}+2\Lambda\left(\frac{du}{dy}\right)^3\right]_{y=1}$",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005, f"Da={Da}, Gr={Gr}, K={K}, $\\Omega_3$={Om3}, $S_n$={S_n}, "
         f"$\\Omega$={Om}, $E_1$={E1}, $\\Lambda$={Lam}, $\\gamma_1$={g1}, $\\lambda_1$={l1}",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
fig.savefig(out("fig3_shear.png"), dpi=170); plt.close(fig)

# ---- combined 3x2 figure: profiles + wall shear + Nusselt, all in one ----
fig, axes = plt.subplots(3, 2, figsize=(13, 15))
for m in M_VALUES:
    r = results[m]
    axes[0, 0].plot(r["y"], r["u"],  styles[m], color=colors[m], lw=2, label=M_LABELS[m])
    axes[0, 1].plot(r["y"], r["th"], styles[m], color=colors[m], lw=2, label=M_LABELS[m])
    Brs, Sxys = shear_Br[m]
    axes[1, 0].plot(Brs, Sxys, mstyles[m], color=colors[m], lw=2, ms=5, label=M_LABELS[m])
    Gs, SxyG = shear_G[m]
    axes[1, 1].plot(Gs, SxyG, mstyles[m], color=colors[m], lw=2, ms=5, label=M_LABELS[m])
    BrsN, Nus = nusselt_Br[m]
    axes[2, 0].plot(BrsN, Nus, mstyles[m], color=colors[m], lw=2, ms=5, label=M_LABELS[m])
    lams, NusL = nusselt_lam[m]
    axes[2, 1].plot(lams, NusL, mstyles[m], color=colors[m], lw=2, ms=5, label=M_LABELS[m])
axes[0, 0].set_title("(a) velocity u(y)")
axes[0, 0].set_xlabel("y"); axes[0, 0].set_ylabel("u")
axes[0, 1].set_title(r"(b) temperature $\theta$(y)")
axes[0, 1].set_xlabel("y"); axes[0, 1].set_ylabel(r"$\theta$")
axes[1, 0].set_title(r"(c) wall shear stress $S_{xy}(y=1)$ vs Br, Eq. (22)")
axes[1, 0].set_xlabel("Brinkman number  Br"); axes[1, 0].set_ylabel(r"$S_{xy}(y=1)$")
axes[1, 1].set_title(r"(d) wall shear stress $S_{xy}(y=1)$ vs G, Eq. (22)")
axes[1, 1].set_xlabel("pressure-gradient parameter  G"); axes[1, 1].set_ylabel(r"$S_{xy}(y=1)$")
axes[2, 0].set_title(r"(e) Nusselt number vs Br, Eq. (23)   ($\lambda$=1)")
axes[2, 0].set_xlabel("Brinkman number  Br"); axes[2, 0].set_ylabel("Nu(y=1)")
axes[2, 1].set_title(r"(f) Nusselt number vs $\lambda$, Eq. (23)   (Br=0.5)")
axes[2, 1].set_xlabel(r"Frank-Kamenetskii parameter  $\lambda$"); axes[2, 1].set_ylabel("Nu(y=1)")
for a in axes.flat:
    a.grid(alpha=0.3); a.legend(fontsize=9)
fig.suptitle("Solution of Eqs. (16)-(17) with BCs (18)-(21):  profiles, "
             "wall shear stress (Eq. 22) and Nusselt number (Eq. 23)",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.005, PSTR, ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.015, 1, 0.97])
fig.savefig(out("fig5_combined.png"), dpi=170); plt.close(fig)


# ======================================================================
# PART 3 : save results to CSV and .npy
# ======================================================================
def mtag(m):
    return f"m={m:g}"


# --- profiles CSV: y, then u/u'/theta/theta' for each m ---
with open(out("results_profiles.csv"), "w", newline="") as f:
    w = csv.writer(f)
    header = ["y"]
    for m in M_VALUES:
        header += [f"u ({mtag(m)})", f"du/dy ({mtag(m)})",
                   f"theta ({mtag(m)})", f"dtheta/dy ({mtag(m)})",
                   f"S_xy ({mtag(m)})"]
    w.writerow(header)
    yy = results[M_VALUES[0]]["y"]
    for i in range(yy.size):
        row = [yy[i]]
        for m in M_VALUES:
            r = results[m]
            row += [r["u"][i], r["up"][i], r["th"][i], r["q"][i],
                    r["up"][i] + 2.0*Lam*r["up"][i]**3]
        w.writerow([f"{v:.10e}" for v in row])

# --- summary CSV ---
with open(out("results_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "u_max", "theta_max", "S_xy(y=1)", "Nu(y=1)"])
    for m in M_VALUES:
        r = results[m]
        w.writerow([m, f"{r['u'].max():.10e}", f"{r['th'].max():.10e}",
                    f"{r['Sxy']:.10e}", f"{r['Nu']:.10e}"])

# --- Nusselt sweeps CSV ---
with open(out("results_nusselt_vs_Br.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "Br", "Nu(y=1)"])
    for m in M_VALUES:
        Brs, Nus = nusselt_Br[m]
        for b, n in zip(Brs, Nus):
            w.writerow([m, f"{b:.4f}", f"{n:.10e}"])

with open(out("results_nusselt_vs_lambda.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "lambda", "Nu(y=1)"])
    for m in M_VALUES:
        lams, Nus = nusselt_lam[m]
        for l, n in zip(lams, Nus):
            w.writerow([m, f"{l:.4f}", f"{n:.10e}"])

# --- wall-shear sweeps CSV (Eq. 22) ---
with open(out("results_shear_vs_Br.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "Br", "S_xy(y=1)"])
    for m in M_VALUES:
        Brs, Sxys = shear_Br[m]
        for b, s in zip(Brs, Sxys):
            w.writerow([m, f"{b:.4f}", f"{s:.10e}"])

with open(out("results_shear_vs_G.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "G", "S_xy(y=1)"])
    for m in M_VALUES:
        Gs, Sxys = shear_G[m]
        for g, s in zip(Gs, Sxys):
            w.writerow([m, f"{g:.4f}", f"{s:.10e}"])

# --- .npy archives (dicts; load with np.load(f, allow_pickle=True).item()) ---
np.save(out("results_profiles.npy"),
        {m: {k: results[m][k] for k in ("y", "u", "up", "th", "q", "Sxy", "Nu")}
         for m in M_VALUES}, allow_pickle=True)
np.save(out("results_nusselt_vs_Br.npy"),
        {m: {"Br": nusselt_Br[m][0], "Nu": nusselt_Br[m][1]} for m in M_VALUES},
        allow_pickle=True)
np.save(out("results_nusselt_vs_lambda.npy"),
        {m: {"lambda": nusselt_lam[m][0], "Nu": nusselt_lam[m][1]} for m in M_VALUES},
        allow_pickle=True)

# --- same npz as the original script, for compatibility ---
np.savez(out("solution_data.npz"),
         **{f"m_{i}": np.vstack([results[m]['y'], results[m]['u'], results[m]['th']])
            for i, m in enumerate(M_VALUES)})

print("\nSaved figures : fig1_velocity.png ... fig5_combined.png, fig8_nusselt.png")
print("Saved CSV     : results_profiles.csv, results_summary.csv,")
print("                results_nusselt_vs_Br.csv, results_nusselt_vs_lambda.csv")
print("Saved NumPy   : results_profiles.npy, results_nusselt_vs_Br.npy,")
print("                results_nusselt_vs_lambda.npy, solution_data.npz")
