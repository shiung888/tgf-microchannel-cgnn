"""Check the claimed failure directions for Da, K and Lambda, and what drives them."""
import numpy as np, warnings
from scipy.integrate import solve_bvp
warnings.filterwarnings("ignore")

P = dict(Lam=0.5, G=2.0, Da=2.0, Gr=3.5, K=2.0, Om3=1.0, Br=0.5,
         S_n=1.0, lam0=1.0, Om=1.0, E1=2.0, g1=0.01, l1=0.01)
m = 0.0

def build(p):
    Lam, G, Da, Gr, K = p["Lam"], p["G"], p["Da"], p["Gr"], p["K"]
    Om3, Br, S_n, lam0 = p["Om3"], p["Br"], p["S_n"], p["lam0"]
    Om, E1, g1, l1 = p["Om"], p["E1"], p["g1"], p["l1"]
    phi = lambda y: np.cosh(K*y)/np.cosh(K)
    def rhs(y, Y):
        u, q, th, r = Y
        arg = np.maximum(1.0 + Om*th, 1e-12)
        upp = -(G - u/Da + Gr*th + K**2*Om3*phi(y))/(1.0 + 6.0*Lam*q**2)
        rpp = -(Br*q**2 + 2.0*Lam*Br*q**4 + S_n + lam0*arg**m*np.exp(-E1/arg))
        return np.vstack([q, upp, r, rpp])
    def bcs(Ya, Yb):
        return np.array([Ya[0] - g1*(Ya[1] + 2*Lam*Ya[1]**3),
                         Yb[0] + g1*(Yb[1] + 2*Lam*Yb[1]**3) - 1.0,
                         Ya[2] - l1*Ya[3], Yb[2] + l1*Yb[3] - 1.0])
    return rhs, bcs

def sweep(name, values):
    print(f"\n--- sweeping {name} from baseline {P[name]} ---")
    print(f"{name:>8}{'ok':>5}{'u_max':>9}{'|u1|max':>9}{'th_max':>9}"
          f"{'visc int':>10}{'3rd-gr int':>11}{'total dissip':>13}")
    y = np.linspace(-1, 1, 401); Y = np.zeros((4, y.size))
    Y[0] = 0.5*(1+y); Y[2] = 0.5*(1+y)
    p = dict(P)
    for v in values:
        p[name] = v
        rhs, bcs = build(p)
        sol = solve_bvp(rhs, bcs, y, Y, tol=1e-8, max_nodes=80000)
        ok = sol.success and np.isfinite(sol.y).all() and abs(sol.y[2]).max() < 5e3
        if not ok:
            print(f"{v:>8.3f}{'NO':>5}   <-- no steady solution found")
            break
        y, Y = sol.x, sol.y
        yy = np.linspace(-1, 1, 8001)
        u, q, th, r = sol.sol(yy)
        vi = np.trapezoid(p["Br"]*q**2, yy)
        ti = np.trapezoid(2*p["Lam"]*p["Br"]*q**4, yy)
        print(f"{v:>8.3f}{'yes':>5}{u.max():>9.3f}{np.abs(q).max():>9.3f}"
              f"{th.max():>9.3f}{vi:>10.3f}{ti:>11.3f}{vi+ti:>13.3f}")

sweep("Da",  [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0])
sweep("K",   [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
sweep("Lam", [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])

sweep("Om3", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
sweep("Da",  [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 1e3, 1e6])
