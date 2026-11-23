"""True turning points (thermal runaway limits) of every swept parameter.

A sweep that stops at a chosen range end cannot say whether a limit lies just
outside it. Here the wall Nusselt number is prescribed and the parameter is an
unknown of the BVP, so the branch can be followed around the fold.

Writes results/critical_points.csv and figures/critical_points.png.
"""
import csv
import os
import warnings
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tgf_model import BASE, FIGURES, M_VALUES, RESULTS, bcs, params, rhs, solve

KEYS = ["Br", "lam", "G", "Gr", "Da", "Lam", "Om3"]
LABEL = dict(Br="Br", lam=r"$\lambda$", G="G", Gr="Gr", Da="Da", Lam=r"$\Lambda$",
             Om3=r"$\Omega_3$")


def trace(job):
    m, key = job
    warnings.filterwarnings("ignore")
    p = params(m=m)
    s = solve(p)
    y, Y, v, Nu = s.x, s.y, p[key], -s.sol(1.0)[3]
    sgn = -1 if key == "Lam" else 1          # Lam becomes critical from above
    pts, step = [(Nu, v)], 0.02
    for _ in range(900):
        N2 = Nu * (1 + step)
        t = solve_bvp(lambda x, Z, q: rhs(x, Z, dict(p, **{key: q[0]})),
                      lambda a, b, q: np.append(bcs(a, b, dict(p, **{key: q[0]})), b[3] + N2),
                      y, Y, p=[v], tol=1e-6, max_nodes=20000)
        if not t.success:
            step /= 2
            if step < 1e-3:
                break
            continue
        y, Y, v, Nu = t.x, t.y, t.p[0], N2
        pts.append((Nu, v))
        step = min(step * 1.2, 0.02)
        best = max(sgn * q[1] for q in pts)
        if sgn * v < best - 0.005 * abs(best) or v > 60 * BASE[key]:
            break
    P = np.array(pts)
    i = int(np.argmax(sgn * P[:, 1]))
    return m, key, P[i, 1], P[i, 0], bool(0 < i < len(P) - 1)


if __name__ == "__main__":
    with Pool() as pool:
        out = pool.map(trace, [(m, k) for m in M_VALUES for k in KEYS], chunksize=1)
    with open(os.path.join(RESULTS, "critical_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "parameter", "baseline", "critical_value", "distance_percent",
                    "Nu_at_fold", "fold_found"])
        for m, k, v, nu, fold in out:
            d = abs(v / BASE[k] - 1) * 100
            w.writerow([m, k, BASE[k], f"{v:.4f}" if fold else "", f"{d:.1f}" if fold else "",
                        f"{nu:.2f}" if fold else "", fold])
            print(f"m={m:>4} {k:<4} baseline {BASE[k]:<4} critical {v:.4f} ({d:.1f}% away)"
                  if fold else f"m={m:>4} {k:<4} no fold found up to {v:.3g}")
    sel = sorted((abs(v / BASE[k] - 1) * 100, k) for m, k, v, n, f in out if m == 0.5 and f)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([LABEL[k] for _, k in sel], [d for d, _ in sel], color="#2a6f97", height=0.6)
    for i, (d, _) in enumerate(sel):
        ax.text(d + 0.4, i, f"{d:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 37)
    ax.set_xlabel("change from baseline that reaches the turning point (%)")
    ax.set_title("Distance to thermal runaway at m = 0.5")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "critical_points.png"), dpi=160)
