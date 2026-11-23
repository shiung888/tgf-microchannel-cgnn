"""Training and test data from the BVP solver, written at full precision.

ofat_train.csv : one factor at a time sweeps x 3 values of m (original design)
lhs_train.csv  : Latin hypercube samples filling the same box
lhs_test.csv   : independent Latin hypercube samples, never used in training

Open these files read-only in Excel: saving them there rounds the numbers.
"""
import csv
import os
import warnings
from multiprocessing import Pool
import numpy as np
from scipy.stats import qmc
from tgf_model import DATA, INPUTS, M_VALUES, RANGES, params, solve, wall_values

N_PER_PARAM, N_LHS_TRAIN, N_LHS_TEST = 60, 1200, 400


def sweep(job):
    warnings.filterwarnings("ignore")
    m, key = job
    rows, guess, streak, tried = [], None, 0, 0
    grid = np.linspace(*RANGES[key], N_PER_PARAM)
    for v in grid:
        p = params(m=m, **{key: v})
        tried += 1
        s = solve(p, guess)
        if s is None and guess is not None:
            s = solve(p)                       # one cold restart
        if s is None:
            guess, streak = None, streak + 1
            if rows and streak >= 5:
                break
            continue
        guess, streak = (s.x, s.y), 0
        rows.append([key] + [p[k] for k in INPUTS] + list(wall_values(s, p)))
    return rows, (m, key, len(rows), tried - len(rows), len(grid) - tried)


def single(x):
    warnings.filterwarnings("ignore")
    p = params(**dict(zip(INPUTS, x)))
    s = solve(p)
    return None if s is None else ["lhs"] + list(x) + list(wall_values(s, p))


def lhs(n, seed):
    U = qmc.LatinHypercube(d=8, seed=seed).random(n)
    lo = np.array([RANGES[k][0] for k in INPUTS[1:]])
    hi = np.array([RANGES[k][1] for k in INPUTS[1:]])
    M = np.array(M_VALUES)[np.minimum((U[:, 0] * 3).astype(int), 2)]
    return np.column_stack([M, lo + U[:, 1:] * (hi - lo)])


def write(name, rows):
    with open(os.path.join(DATA, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["varied"] + INPUTS + ["Sxy_wall", "Nu_wall"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.10g}" for v in r[1:]])
    print(f"{name}: {len(rows)} samples")


if __name__ == "__main__":
    with Pool() as pool:
        out = pool.map(sweep, [(m, k) for m in M_VALUES for k in RANGES], chunksize=1)
        for _, (m, k, ok, fail, skip) in out:
            if fail or skip:
                print(f"m={m:>4} sweep {k:<4}: {ok} converged, {fail} failed, "
                      f"{skip} skipped after 5 failures in a row")
        write("ofat_train.csv", [r for rows, _ in out for r in rows])
        for name, n, seed in (("lhs_train.csv", N_LHS_TRAIN, 101),
                              ("lhs_test.csv", N_LHS_TEST, 202)):
            res = pool.map(single, lhs(n, seed), chunksize=8)
            print(f"{name}: no steady solution at {sum(r is None for r in res)} of {n} points")
            write(name, [r for r in res if r is not None])
