"""CG surrogate trained on each design and scored on three test sets. 10 seeds.

own held-out split : random 15% of the same design (interpolation only)
independent LHS    : 400 points spread over the whole box
OFAT sweeps        : the sweeps, which include the steep region near runaway
                     (only meaningful for the design that never saw them)
MAPE is not useful for Sxy on LHS points, because Sxy crosses zero there.
Writes results/surrogate_*.csv.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import csv
from multiprocessing import Pool
import numpy as np
from network import Scaler, forward, init, load, metrics, split, train_cg
from tgf_model import RESULTS

SEEDS = range(1, 11)
DESIGNS = {"OFAT": ["ofat_train.csv"], "LHS": ["lhs_train.csv"],
           "OFAT+LHS": ["ofat_train.csv", "lhs_train.csv"]}
TARGETS = ("Nu_wall", "Sxy_wall")


def run(job):
    design, target, seed = job
    parts = [load(f, target) for f in DESIGNS[design]]
    X, t = np.vstack([a for a, _ in parts]), np.concatenate([b for _, b in parts])
    tr, va, te = split(len(t), seed)
    s = Scaler(X[tr], t[tr])
    w, _, _ = train_cg(init(seed), s.x(X[tr]), s.t(t[tr]), s.x(X[va]), s.t(t[va]))
    Xi, ti = load("lhs_test.csv", target)
    tests = [("own held-out split", t[te], X[te]), ("independent LHS", ti, Xi)]
    if design == "LHS":
        Xo, to = load("ofat_train.csv", target)
        tests.append(("OFAT sweeps", to, Xo))
    return [dict(design=design, target=target, seed=seed, tested_on=n,
                 **metrics(a, s.inv(forward(w, s.x(b))[0])))
            for n, a, b in tests]


if __name__ == "__main__":
    jobs = [(d, t, s) for d in DESIGNS for t in TARGETS for s in SEEDS]
    with Pool() as p:
        R = [r for x in p.map(run, jobs, chunksize=1) for r in x]
    with open(os.path.join(RESULTS, "surrogate_all_seeds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, R[0])
        w.writeheader()
        w.writerows(R)
    cols = ["R2", "RMSE", "MAPE", "MedAPE", "MaxAPE"]
    with open(os.path.join(RESULTS, "surrogate_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "target", "tested_on"] + [f"{c}_median" for c in cols]
                   + ["R2_min", "R2_max"])
        for k in sorted({(r["design"], r["target"], r["tested_on"]) for r in R}):
            S = [r for r in R if (r["design"], r["target"], r["tested_on"]) == k]
            row = (list(k) + [f"{np.median([r[c] for r in S]):.4g}" for c in cols]
                   + [f"{min(r['R2'] for r in S):.4g}", f"{max(r['R2'] for r in S):.4g}"])
            w.writerow(row)
            print(*row, sep=" | ")
