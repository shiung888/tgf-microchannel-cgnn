"""
Boundary value problem for electroosmotic flow of a reactive third-grade fluid
in a vertical porous microchannel with velocity and thermal slip.

    u'' + 6 Lam u'^2 u'' + G - u/Da + Gr th + K^2 Om3 phi = 0            (16)
    th'' + Br u'^2 + 2 Lam Br u'^4 + S_n
         + lam (1 + Om th)^m exp(-E1/(1 + Om th)) = 0                    (17)
    phi = cosh(K y)/cosh(K)                                              (14)
    u(-1) = g1 S(-1),   u(1) = 1 - g1 S(1),   S = u' + 2 Lam u'^3    (18, 19)
    th(-1) = l1 th'(-1),   th(1) = 1 - l1 th'(1)                     (20, 21)

Wall quantities: Sxy = S(1) (22) and Nu = -th'(1) (23).
Every script imports the model from here, so there is only one copy of it.
"""
import os
import numpy as np
from scipy.integrate import solve_bvp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
for _d in (DATA, RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)

BASE = dict(m=0.0, Br=0.5, lam=1.0, G=2.0, Gr=3.5, Da=2.0, Lam=0.5, Om3=1.0,
            K=2.0, S_n=1.0, Om=1.0, E1=2.0, g1=0.01, l1=0.01)
M_VALUES = [-2.0, 0.0, 0.5]
INPUTS = ["m", "Br", "lam", "G", "Gr", "Da", "Lam", "Om3"]
RANGES = dict(Br=(0.0, 0.5), lam=(0.0, 1.0), G=(0.0, 2.0), Gr=(0.0, 3.5),
              Da=(0.2, 5.0), Lam=(0.4, 2.0), Om3=(0.0, 2.0))

trapezoid = getattr(np, "trapezoid", None) or np.trapz      # NumPy 1.x and 2.x


def params(**kw):
    p = dict(BASE)
    p.update(kw)
    return p


def phi(y, K):
    return np.cosh(K * y) / np.cosh(K)


def rhs(y, Y, p):
    u, du, th, dth = Y
    arg = np.maximum(1.0 + p["Om"] * th, 1e-12)
    react = p["lam"] * arg**p["m"] * np.exp(-p["E1"] / arg)
    d2u = -(p["G"] - u / p["Da"] + p["Gr"] * th
            + p["K"]**2 * p["Om3"] * phi(y, p["K"])) / (1 + 6 * p["Lam"] * du**2)
    d2th = -(p["Br"] * du**2 + 2 * p["Lam"] * p["Br"] * du**4 + p["S_n"] + react)
    return np.vstack([du, d2u, dth, d2th])


def bcs(a, b, p):
    def S(du):
        return du + 2 * p["Lam"] * du**3
    return np.array([a[0] - p["g1"] * S(a[1]), b[0] + p["g1"] * S(b[1]) - 1,
                     a[2] - p["l1"] * a[3], b[2] + p["l1"] * b[3] - 1])


def _attempt(p, y, Y, tol, nodes):
    s = solve_bvp(lambda x, Z: rhs(x, Z, p), lambda a, b: bcs(a, b, p),
                  y, Y, tol=tol, max_nodes=nodes)
    ok = s.success and np.isfinite(s.y).all() and np.abs(s.y[2]).max() < 5e3
    return s if ok else None


def solve(p, guess=None, tol=1e-7):
    """Solve one case. Cold starts ramp (Br, lam, Gr) together. None = no solution."""
    if guess is None:
        y = np.linspace(-1, 1, 121)
        Y = np.zeros((4, y.size))
        Y[0] = Y[2] = 0.5 * (1 + y)
        ramp = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
    else:
        (y, Y), ramp = guess, (1.0,)
    for f in ramp:
        q = dict(p, Br=f * p["Br"], lam=f * p["lam"], Gr=f * p["Gr"])
        s = _attempt(q, y, Y, 1e-6, 6000)
        if s is None:
            return None
        y, Y = s.x, s.y
    return _attempt(p, y, Y, tol, 120000 if tol < 1e-8 else 20000)


def wall_values(s, p):
    du, dth = s.sol(1.0)[1], s.sol(1.0)[3]
    return du + 2 * p["Lam"] * du**3, -dth
