"""Solver verification: ODE residuals, boundary residuals and the Newtonian limit.

Writes results/verification.csv and results/baseline_summary.csv.
"""
import csv
import os
import numpy as np
from scipy.integrate import solve_bvp
from tgf_model import M_VALUES, RESULTS, bcs, params, phi, rhs, solve, wall_values

rows, summ = [], []
yy = np.linspace(-1, 1, 4001)
for m in M_VALUES:
    p = params(m=m)
    s = solve(p, tol=1e-9)
    Z, dZ = s.sol(yy), s.sol(yy, 1)
    u, du, th, dth = Z
    arg = 1 + p["Om"] * th
    r16 = (dZ[1] * (1 + 6 * p["Lam"] * du**2) + p["G"] - u / p["Da"] + p["Gr"] * th
           + p["K"]**2 * p["Om3"] * phi(yy, p["K"]))
    r17 = (dZ[3] + p["Br"] * du**2 + 2 * p["Lam"] * p["Br"] * du**4 + p["S_n"]
           + p["lam"] * arg**m * np.exp(-p["E1"] / arg))
    rb = bcs(s.sol(-1.0), s.sol(1.0), p)
    rows.append([f"m={m:g}", "max |Eq16|", f"{np.abs(r16).max():.2e}"])
    rows.append([f"m={m:g}", "max |Eq17|", f"{np.abs(r17).max():.2e}"])
    rows.append([f"m={m:g}", "max |BC|", f"{np.abs(rb).max():.2e}"])
    sxy, nu = wall_values(s, p)
    summ.append([m, f"{u.max():.6f}", f"{th.max():.6f}", f"{sxy:.6f}", f"{nu:.6f}"])

# Newtonian limit: Lam = Br = lam = S_n = Gr = g1 = 0, exact solution known
p = params(Lam=0.0, Br=0.0, lam=0.0, S_n=0.0, Gr=0.0, g1=0.0)
y = np.linspace(-1, 1, 201)
Y = np.vstack([.5 * (1 + y), .5 + 0 * y, .5 * (1 + y), .5 + 0 * y])
s = solve_bvp(lambda x, Z: rhs(x, Z, p), lambda a, b: bcs(a, b, p), y, Y, tol=1e-9,
              max_nodes=120000)
a, K = 1 / np.sqrt(p["Da"]), p["K"]
C = p["G"] * p["Da"]
D = -K**2 * p["Om3"] / (np.cosh(K) * (K**2 - 1 / p["Da"]))
B = 1 / (2 * np.sinh(a))
A = (0.5 - C - D * np.cosh(K)) / np.cosh(a)
ex = A * np.cosh(a * yy) + B * np.sinh(a * yy) + C + D * np.cosh(K * yy)
dex = A * a * np.sinh(a) + B * a * np.cosh(a) + D * K * np.sinh(K)
rows.append(["Newtonian limit", "max |u - exact|", f"{np.abs(s.sol(yy)[0] - ex).max():.2e}"])
rows.append(["Newtonian limit", "|Sxy - exact|", f"{abs(s.sol(1.0)[1] - dex):.2e}"])

for name, head, body in (("verification.csv", ["case", "check", "value"], rows),
                         ("baseline_summary.csv",
                          ["m", "u_max", "theta_max", "Sxy_wall", "Nu_wall"], summ)):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(body)
for r in rows + summ:
    print(*r)
