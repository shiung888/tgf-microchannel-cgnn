"""Check whether the reported max residual depends on the 4001-point sampling choice."""
import numpy as np, warnings
from scipy.integrate import solve_bvp
warnings.filterwarnings("ignore")

Lam, G, Da, Gr, K, Om3, Br = 0.5, 2.0, 2.0, 3.5, 2.0, 1.0, 0.5
S_n, lam0, Om, E1, g1, l1 = 1.0, 1.0, 1.0, 2.0, 0.01, 0.01
M_VALUES = [-2.0, 0.0, 0.5]

def phi(y): return np.cosh(K*y)/np.cosh(K)

def make_system(m, BrX, lamX):
    def rhs(y, Y):
        u, p, th, q = Y
        arg = np.maximum(1.0 + Om*th, 1e-12)
        react = lamX*arg**m*np.exp(-E1/arg)
        upp = -(G - u/Da + Gr*th + K**2*Om3*phi(y))/(1.0 + 6.0*Lam*p**2)
        qpp = -(BrX*p**2 + 2.0*Lam*BrX*p**4 + S_n + react)
        return np.vstack([p, upp, q, qpp])
    def bcs(Ya, Yb):
        return np.array([
            Ya[0] - g1*(Ya[1] + 2.0*Lam*Ya[1]**3),
            Yb[0] + g1*(Yb[1] + 2.0*Lam*Yb[1]**3) - 1.0,
            Ya[2] - l1*Ya[3],
            Yb[2] + l1*Yb[3] - 1.0,
        ])
    return rhs, bcs

def solve_for_m(m):
    y = np.linspace(-1.0, 1.0, 201)
    Y = np.zeros((4, y.size)); Y[0] = 0.5*(1+y); Y[2] = 0.5*(1+y)
    for BrX in [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, Br]:
        rhs, bcs = make_system(m, BrX, lam0)
        sol = solve_bvp(rhs, bcs, y, Y, tol=1e-6, max_nodes=60000)
        y, Y = sol.x, sol.y
    rhs, bcs = make_system(m, Br, lam0)
    return solve_bvp(rhs, bcs, y, Y, tol=1e-9, max_nodes=120000)

def residuals(sol, yv, m):
    u, p, th, q = sol.sol(yv)
    dY = sol.sol(yv, 1); upp, qpp = dY[1], dY[3]
    arg = 1.0 + Om*th
    r16 = upp + 6*Lam*p**2*upp + G - u/Da + Gr*th + K**2*Om3*phi(yv)
    r17 = qpp + Br*p**2 + 2*Lam*Br*p**4 + S_n + lam0*arg**m*np.exp(-E1/arg)
    return np.abs(r16).max(), np.abs(r17).max()

COUNTS = [501, 1001, 2001, 4001, 8001, 16001, 40001, 100001]
for m in M_VALUES:
    sol = solve_for_m(m)
    nodes = sol.x.size
    gaps = np.diff(sol.x)
    print(f"\nm = {m}   solver converged with {nodes} mesh nodes "
          f"({nodes-1} subintervals), spacing min {gaps.min():.2e} max {gaps.max():.2e}")
    print(f"   uniform 4001 spacing = {2/4000:.2e}   "
          f"ratio to coarsest solver gap = {gaps.max()/(2/4000):.1f}x finer")
    # residual sampled ON the solver's own mesh nodes vs off it
    rn = residuals(sol, sol.x, m)
    print(f"   sampled at solver's own nodes : |res16|={rn[0]:.3e}  |res17|={rn[1]:.3e}")
    print("   sampled on uniform grids")
    for n in COUNTS:
        yv = np.linspace(-1, 1, n)
        a, b = residuals(sol, yv, m)
        tag = "  <-- as reported" if n == 4001 else ""
        print(f"      n={n:>6}   |res16|={a:.6e}   |res17|={b:.6e}{tag}")
    # also an even count, to test the odd/even question
    for n in [4000, 2000]:
        yv = np.linspace(-1, 1, n)
        a, b = residuals(sol, yv, m)
        print(f"      n={n:>6}   |res16|={a:.6e}   |res17|={b:.6e}   (even)")
