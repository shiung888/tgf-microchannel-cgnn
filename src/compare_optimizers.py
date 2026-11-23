"""CG against Adam (batch, mini-batch, SGD): same w0, same split, 10 seeds.

The learning rate of each tuned Adam regime is picked by VALIDATION error, never
by test error. Tuned batch Adam also gets about the same number of full gradient
evaluations as CG (5500). The settings of the first study (lr = 1e-3, 1500
epochs) are kept as separate rows. Writes results/optimizers_*.csv.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import csv
from multiprocessing import Pool
import numpy as np
from network import Scaler, forward, init, load, metrics, split, train_adam, train_cg
from tgf_model import RESULTS

SEEDS = range(1, 11)
DESIGNS = {"OFAT": "ofat_train.csv", "LHS": "lhs_train.csv"}
TARGETS = ("Nu_wall", "Sxy_wall")
# method -> (batch size, epochs, learning rates tried)
METHODS = {
    "CG": None,
    "Batch Adam, first study": (None, 1500, [1e-3]),
    "Batch Adam, tuned": (None, 5500, [1e-3, 1e-2, 3e-2]),
    "Mini-batch Adam, first study": (32, 1500, [1e-3]),
    "Mini-batch Adam, tuned": (32, 1500, [1e-3, 3e-3, 1e-2]),
    "SGD Adam, first study": (1, 1500, [1e-3]),
}


def run(job):
    design, target, seed, name = job
    X, t = load(DESIGNS[design], target)
    tr, va, te = split(len(t), seed)
    s = Scaler(X[tr], t[tr])
    parts = (s.x(X[tr]), s.t(t[tr]), s.x(X[va]), s.t(t[va]))
    w0, lr = init(seed), ""
    if METHODS[name] is None:
        w, v, n = train_cg(w0, *parts)
        samples = n * len(tr)
    else:
        batch, epochs, rates = METHODS[name]
        tried = [(train_adam(w0, *parts, batch, epochs, q, seed), q) for q in rates]
        (w, v, n), lr = min(tried, key=lambda r: r[0][1])
        samples = epochs * len(tr)
    Xi, ti = load("lhs_test.csv", target)
    a = metrics(t[te], s.inv(forward(w, s.x(X[te]))[0]))
    c = metrics(ti, s.inv(forward(w, s.x(Xi))[0]))
    return dict(design=design, target=target, method=name, seed=seed, lr=lr,
                gradient_calls=n, sample_gradients=samples, R2=a["R2"], RMSE=a["RMSE"],
                MAPE=a["MAPE"], R2_independent=c["R2"], RMSE_independent=c["RMSE"])


if __name__ == "__main__":
    jobs = [(d, t, s, m) for m in list(METHODS)[::-1] for d in DESIGNS
            for t in TARGETS for s in SEEDS]
    with Pool() as p:
        R = p.map(run, jobs, chunksize=1)
    with open(os.path.join(RESULTS, "optimizers_all_seeds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, R[0])
        w.writeheader()
        w.writerows(R)
    with open(os.path.join(RESULTS, "optimizers_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "target", "method", "R2_mean", "RMSE_mean", "RMSE_std",
                    "MAPE_mean", "RMSE_independent_mean", "CG_wins_of_10",
                    "CG_wins_independent_of_10", "sample_gradients_mean"])
        for d in DESIGNS:
            for t in TARGETS:
                def pick(m):
                    return [r for r in R if (r["design"], r["target"], r["method"]) == (d, t, m)]
                cg = {r["seed"]: r for r in pick("CG")}
                for m in METHODS:
                    S = pick(m)

                    def mean(c):
                        return np.mean([r[c] for r in S])

                    def wins(c):
                        return "" if m == "CG" else sum(cg[r["seed"]][c] < r[c] for r in S)
                    row = [d, t, m, f"{mean('R2'):.4f}", f"{mean('RMSE'):.4f}",
                           f"{np.std([r['RMSE'] for r in S], ddof=1):.4f}",
                           f"{mean('MAPE'):.2f}", f"{mean('RMSE_independent'):.4f}",
                           wins("RMSE"), wins("RMSE_independent"),
                           f"{mean('sample_gradients'):.3g}"]
                    w.writerow(row)
                    print(*row, sep=" | ")
