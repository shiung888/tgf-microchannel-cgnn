"""Shared surrogate code: one network, one split, one initial weight vector.

Every optimiser receives the SAME w0 and the SAME split for a given seed. They
come from separate seeded generators, so neither depends on the order of calls.
"""
import os
import numpy as np
from scipy.optimize import minimize
from tgf_model import DATA, INPUTS

H, D = 10, len(INPUTS)
NW = D * H + H + H + 1


def load(name, target):
    A = np.genfromtxt(os.path.join(DATA, name), delimiter=",", names=True, dtype=None,
                      encoding="utf-8")
    return np.column_stack([A[k] for k in INPUTS]).astype(float), A[target].astype(float)


def split(n, seed):
    perm = np.random.default_rng([seed, 1]).permutation(n)
    a, b = int(0.70 * n), int(0.15 * n)
    return perm[:a], perm[a:a + b], perm[a + b:]


def init(seed):
    return np.random.default_rng([seed, 2]).normal(0.0, 0.5, NW)


class Scaler:
    """Map inputs and target to [-1, 1] using training data only."""

    def __init__(self, X, t):
        self.lo, self.hi, self.a, self.b = X.min(0), X.max(0), t.min(), t.max()

    def x(self, X):
        return 2 * (X - self.lo) / (self.hi - self.lo) - 1

    def t(self, t):
        return 2 * (t - self.a) / (self.b - self.a) - 1

    def inv(self, t):
        return (t + 1) / 2 * (self.b - self.a) + self.a


def forward(w, X):
    h = np.tanh(X @ w[:D * H].reshape(D, H) + w[D * H:D * H + H])
    return h @ w[D * H + H:D * H + 2 * H] + w[-1], h


def loss_grad(w, X, t):
    o, h = forward(w, X)
    e = o - t
    g = 2 * e / len(t)
    gh = np.outer(g, w[D * H + H:D * H + 2 * H]) * (1 - h * h)
    return np.mean(e * e), np.concatenate([(X.T @ gh).ravel(), gh.sum(0), h.T @ g, [g.sum()]])


def train_cg(w0, Xt, tt, Xv, tv, maxiter=3000):
    """Nonlinear CG (SciPy: Polak-Ribiere with max(beta, 0), strong Wolfe line search)."""
    best = {"v": np.inf, "w": w0}

    def cb(w):
        v = np.mean((forward(w, Xv)[0] - tv)**2)
        if v < best["v"]:
            best.update(v=v, w=w.copy())

    r = minimize(loss_grad, w0, args=(Xt, tt), jac=True, method="CG", callback=cb,
                 options=dict(maxiter=maxiter, gtol=1e-10))
    return best["w"], best["v"], r.njev


def train_adam(w0, Xt, tt, Xv, tv, batch, epochs, lr, seed):
    w, m, v = w0.copy(), np.zeros(NW), np.zeros(NW)
    rng, N, k = np.random.default_rng([seed, 3]), len(tt), 0
    batch = batch or N
    bv, bw = np.inf, w
    for _ in range(epochs):
        o = rng.permutation(N)
        for i in range(0, N, batch):
            j = o[i:i + batch]
            g = loss_grad(w, Xt[j], tt[j])[1]
            k += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            w = w - lr * (m / (1 - 0.9**k)) / (np.sqrt(v / (1 - 0.999**k)) + 1e-8)
        e = np.mean((forward(w, Xv)[0] - tv)**2)
        if e < bv:
            bv, bw = e, w.copy()
    return bw, bv, k


def metrics(t, y):
    e = t - y
    return dict(R2=1 - (e @ e) / ((t - t.mean())**2).sum(), RMSE=np.sqrt(np.mean(e * e)),
                MAPE=100 * np.mean(np.abs(e / t)), MedAPE=100 * np.median(np.abs(e / t)),
                MaxAPE=100 * np.max(np.abs(e / t)))
