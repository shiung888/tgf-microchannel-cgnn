"""Why does u peak right of centre while theta peaks left of centre?"""
import numpy as np, warnings
from scipy.integrate import solve_bvp
warnings.filterwarnings("ignore")
trapezoid = getattr(np, "trapezoid", None) or np.trapz   # NumPy 1.x and 2.x

Lam, G, Da, Gr, K, Om3, Br = 0.5, 2.0, 2.0, 3.5, 2.0, 1.0, 0.5
S_n, lam0, Om, E1, g1, l1 = 1.0, 1.0, 1.0, 2.0, 0.01, 0.01
M = [-2.0, 0.0, 0.5]

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
        return np.array([Ya[0] - g1*(Ya[1] + 2.0*Lam*Ya[1]**3),
                         Yb[0] + g1*(Yb[1] + 2.0*Lam*Yb[1]**3) - 1.0,
                         Ya[2] - l1*Ya[3],
                         Yb[2] + l1*Yb[3] - 1.0])
    return rhs, bcs

def solve_for_m(m, BrX=Br, lamX=lam0, GrX=None):
    global Gr
    if GrX is not None: Gr = GrX
    y = np.linspace(-1.0, 1.0, 201); Y = np.zeros((4, y.size))
    Y[0] = 0.5*(1+y); Y[2] = 0.5*(1+y)
    for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, BrX]:
        rhs, bcs = make_system(m, b, lamX)
        sol = solve_bvp(rhs, bcs, y, Y, tol=1e-6, max_nodes=60000)
        y, Y = sol.x, sol.y
    rhs, bcs = make_system(m, BrX, lamX)
    return solve_bvp(rhs, bcs, y, Y, tol=1e-9, max_nodes=120000)

yy = np.linspace(-1, 1, 20001)
print("PEAK LOCATIONS AND WALL SLOPES")
print(f"{'m':>6}{'y*(u peak)':>12}{'y*(th peak)':>13}{'u(-1)':>9}{'u(+1)':>9}"
      f"{'th(-1)':>9}{'th(+1)':>9}{'u1(-1)':>10}{'u1(+1)':>10}")
sols = {}
for m in M:
    s = solve_for_m(m); sols[m] = s
    u, p, th, q = s.sol(yy)
    print(f"{m:>6}{yy[u.argmax()]:>12.4f}{yy[th.argmax()]:>13.4f}"
          f"{u[0]:>9.4f}{u[-1]:>9.4f}{th[0]:>9.4f}{th[-1]:>9.4f}"
          f"{p[0]:>10.4f}{p[-1]:>10.4f}")

print("\nHEAT SOURCE SPLIT  (integral of each term over left half vs right half)")
print(f"{'m':>6}  {'term':<22}{'left':>10}{'right':>10}{'L/R':>8}")
for m in M:
    u, p, th, q = sols[m].sol(yy)
    arg = 1.0 + Om*th
    terms = {"viscous Br*u1^2": Br*p**2,
             "third grade 2LBr*u1^4": 2*Lam*Br*p**4,
             "Joule S_n": np.full_like(yy, S_n),
             "reaction": lam0*arg**m*np.exp(-E1/arg)}
    L = yy < 0
    for name, t in terms.items():
        li = trapezoid(t[L], yy[L]); ri = trapezoid(t[~L], yy[~L])
        print(f"{m:>6}  {name:<22}{li:>10.3f}{ri:>10.3f}{li/ri:>8.2f}")
    tot = sum(terms.values())
    li = trapezoid(tot[L], yy[L]); ri = trapezoid(tot[~L], yy[~L])
    print(f"{m:>6}  {'TOTAL SOURCE':<22}{li:>10.3f}{ri:>10.3f}{li/ri:>8.2f}\n")

print("CONTROL EXPERIMENTS (m = 0)")
print("Does the theta peak move right if dissipation is switched off?")
for b in [0.5, 0.25, 0.1, 0.0]:
    s = solve_for_m(0.0, BrX=b)
    u, p, th, q = s.sol(yy)
    print(f"   Br={b:<5}  y*(u)={yy[u.argmax()]:+.4f}   y*(theta)={yy[th.argmax()]:+.4f}"
          f"   u1(-1)={p[0]:+.3f}  u1(+1)={p[-1]:+.3f}")
print("Does the u peak move if buoyancy is switched off?  (Br restored to 0.5)")
for gr in [3.5, 2.0, 0.0]:
    s = solve_for_m(0.0, GrX=gr)
    u, p, th, q = s.sol(yy)
    print(f"   Gr={gr:<5}  y*(u)={yy[u.argmax()]:+.4f}   y*(theta)={yy[th.argmax()]:+.4f}")
