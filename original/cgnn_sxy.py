"""
CGNN (conjugate-gradient neural network) for predicting the wall shear
stress S_xy(y=1) (Eq. 22), in the style of 11111.pdf (Abd-Elmonem et al.)
and Journal 1 (Soomro et al.).

Network   : feed-forward, 1 hidden layer of N_HIDDEN tanh neurons,
            linear output (tansig + purelin, as in the journals)
Training  : conjugate gradient (scipy.optimize.minimize, Polak-Ribiere CG)
            minimizing MSE on the normalized training set - no learning
            rate needed (step size found by line search, Eq. 28 of the CGNN
            paper)
Split     : 70% training / 15% validation / 15% testing (as Journal 1);
            the weights kept are those with the LOWEST VALIDATION MSE
            (protection against over-fitting)
Data      : ann_training_dataset_ofat.csv by default (8 inputs); works with
            ann_training_dataset.csv (5 inputs) too - inputs are detected
            from the CSV header automatically.

Reported  : MSE, RMSE and R^2 per split (train/validation/test/all),
            denormalized units.

Outputs:
    fig16_cgnn_sxy_performance.png regression scatter (train/val/test/all)
                                   + error histogram + training curve
    fig17_cgnn_sxy_sweeps.png      CGNN prediction vs BVP data along each
                                   OFAT sweep (only if CSV has 'varied')
    cgnn_sxy_results.csv           per-sample prediction, error, split
"""

import os
import csv
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------- settings --------------------------------
CSV_FILE  = "ann_training_dataset_ofat.csv"   # or "ann_training_dataset.csv"
TARGET    = "Sxy_wall"                         # this script predicts S_xy(y=1)
N_HIDDEN  = 10
MAX_ITER  = 3000
SEED      = 7                                  # same seed as cgnn_nu.py -> same split
FIG_PERF  = "fig16_cgnn_sxy_performance.png"
FIG_SWEEP = "fig17_cgnn_sxy_sweeps.png"
OUT_CSV   = "cgnn_sxy_results.csv"
MET_CSV   = "cgnn_sxy_metrics.csv"
TARGET_TEX = "$S_{xy}(y{=}1)$"

# ------------------------------ load data ------------------------------
with open(os.path.join(OUTDIR, CSV_FILE)) as f:
    rdr = list(csv.reader(f))
header, data = rdr[0], rdr[1:]
has_varied = header[0] == "varied"
in_cols = [c for c in header if c not in ("varied", "Sxy_wall", "Nu_wall")]
in_idx = [header.index(c) for c in in_cols]
t_idx = header.index(TARGET)

X = np.array([[float(r[i]) for i in in_idx] for r in data])
t = np.array([float(r[t_idx]) for r in data])
varied = [r[0] for r in data] if has_varied else None
n, d = X.shape
print(f"{CSV_FILE}: {n} samples, {d} inputs {in_cols} -> target {TARGET}")

# --------------------------- 70/15/15 split ----------------------------
rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
n_tr, n_va = int(0.70 * n), int(0.15 * n)
idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
split = np.empty(n, dtype=object)
split[idx_tr], split[idx_va], split[idx_te] = "train", "validation", "test"

# ------------------ normalization (fit on training set) ----------------
# map to [-1, 1], like MATLAB's mapminmax
Xmin, Xmax = X[idx_tr].min(0), X[idx_tr].max(0)
tmin, tmax = t[idx_tr].min(), t[idx_tr].max()
Xn = 2 * (X - Xmin) / (Xmax - Xmin) - 1
tn = 2 * (t - tmin) / (tmax - tmin) - 1


def denorm(yn):
    return (yn + 1) / 2 * (tmax - tmin) + tmin


# --------------------------- CGNN definition ---------------------------
# weights vector w = [W1 (d x H), b1 (H), W2 (H), b2 (1)]
H = N_HIDDEN
n_w = d * H + H + H + 1


def unpack(w):
    i = 0
    W1 = w[i:i + d * H].reshape(d, H); i += d * H
    b1 = w[i:i + H]; i += H
    W2 = w[i:i + H]; i += H
    b2 = w[i]
    return W1, b1, W2, b2


def forward(w, Xin):
    W1, b1, W2, b2 = unpack(w)
    hid = np.tanh(Xin @ W1 + b1)          # tansig hidden layer
    return hid @ W2 + b2, hid             # purelin output


def loss_grad(w, Xin, tgt):
    """MSE loss and its analytic gradient (backprop)."""
    W1, b1, W2, b2 = unpack(w)
    hid = np.tanh(Xin @ W1 + b1)
    out = hid @ W2 + b2
    err = out - tgt
    m = Xin.shape[0]
    L = np.mean(err**2)
    d_out = 2.0 * err / m                          # dL/d(out)
    gW2 = hid.T @ d_out
    gb2 = d_out.sum()
    d_hid = np.outer(d_out, W2) * (1 - hid**2)     # through tanh
    gW1 = Xin.T @ d_hid
    gb1 = d_hid.sum(0)
    return L, np.concatenate([gW1.ravel(), gb1, gW2, [gb2]])


# --------------------------- training (CG) -----------------------------
w0 = rng.normal(0.0, 0.5, n_w)
hist = {"iter": [], "train": [], "val": []}
best = {"w": w0.copy(), "val": np.inf, "iter": 0}


def callback(wk):
    it = len(hist["iter"])
    mtr = loss_grad(wk, Xn[idx_tr], tn[idx_tr])[0]
    mva = loss_grad(wk, Xn[idx_va], tn[idx_va])[0]
    hist["iter"].append(it); hist["train"].append(mtr); hist["val"].append(mva)
    if mva < best["val"]:
        best.update(w=wk.copy(), val=mva, iter=it)


t_start = time.time()
res = minimize(loss_grad, w0, args=(Xn[idx_tr], tn[idx_tr]), jac=True,
               method="CG", callback=callback,
               options={"maxiter": MAX_ITER, "gtol": 1e-10})
train_time_s = time.time() - t_start
w_best = best["w"]
print(f"CG stopped after {len(hist['iter'])} iterations "
      f"({res.message}); best validation MSE at iteration {best['iter']}")
print(f"training time {train_time_s:.2f} s | {res.nfev} function evaluations, "
      f"{res.njev} gradient evaluations (no learning rate required)")

# ----------------------------- evaluation ------------------------------
yn_all = forward(w_best, Xn)[0]
y_all = denorm(yn_all)                 # denormalized predictions
e_all = t - y_all                      # errors in physical units


def metrics(idx):
    """Absolute measures (MSE, RMSE) and relative measures (MAPE, median APE).

    R^2 is the fraction of variance explained, which is not an error measure.
    MAPE expresses the error as a percentage of the true value, which is what
    a designer can act on. Both are reported so the two are not confused.
    """
    err = e_all[idx]
    mse = np.mean(err**2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum(err**2) / np.sum((t[idx] - t[idx].mean())**2)
    denom = np.where(t[idx] == 0.0, np.nan, t[idx])
    ape = np.abs(err / denom) * 100.0          # absolute percentage error
    return mse, rmse, r2, np.nanmean(ape), np.nanmedian(ape)


print(f"\n{'split':<12}{'MSE':>12}{'RMSE':>10}{'R^2':>10}"
      f"{'MAPE %':>10}{'MedAPE %':>11}")
SPLITS = [("train", idx_tr), ("validation", idx_va),
          ("test", idx_te), ("all", np.arange(n))]
rows_metrics = []
for name, idx in SPLITS:
    mse, rmse, r2, mape, medape = metrics(idx)
    print(f"{name:<12}{mse:>12.3e}{rmse:>10.4f}{r2:>10.6f}"
          f"{mape:>10.2f}{medape:>11.2f}")
    rows_metrics.append([name, f"{mse:.6e}", f"{rmse:.4f}", f"{r2:.6f}",
                         f"{mape:.2f}", f"{medape:.2f}", f"{train_time_s:.2f}"])

# ------------------------- metrics summary CSV -------------------------
with open(os.path.join(OUTDIR, MET_CSV), "w", newline="") as f:
    wm = csv.writer(f)
    wm.writerow(["split", "MSE", "RMSE", "R2", "MAPE_pct", "MedAPE_pct",
                 "train_time_s"])
    wm.writerows(rows_metrics)
print(f"Saved {MET_CSV}")

# ------------------- training history and run summary ------------------
HIST_CSV = "cgnn_sxy_history.csv"
with open(os.path.join(OUTDIR, HIST_CSV), "w", newline="") as f:
    wh = csv.writer(f)
    wh.writerow(["iteration", "train_MSE_normalised", "val_MSE_normalised"])
    for it, a, b in zip(hist["iter"], hist["train"], hist["val"]):
        wh.writerow([it, f"{a:.8e}", f"{b:.8e}"])
print(f"Saved {HIST_CSV}")

SUM_CSV = "cgnn_sxy_run_summary.csv"
with open(os.path.join(OUTDIR, SUM_CSV), "w", newline="") as f:
    ws = csv.writer(f)
    ws.writerow(["target", "iterations", "best_val_iteration",
                 "function_evaluations", "gradient_evaluations",
                 "train_time_s", "termination_message"])
    ws.writerow([TARGET, len(hist["iter"]), best["iter"],
                 res.nfev, res.njev, f"{train_time_s:.3f}", res.message])
print(f"Saved {SUM_CSV}  <-- send these numbers for Table 4.7")


# --------------------------- results CSV -------------------------------
with open(os.path.join(OUTDIR, OUT_CSV), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(in_cols + [TARGET, "CGNN_pred", "error", "split"])
    for i in range(n):
        w.writerow([f"{v:.6g}" for v in X[i]] +
                   [f"{t[i]:.8e}", f"{y_all[i]:.8e}", f"{e_all[i]:.3e}", split[i]])
print(f"Saved {OUT_CSV}")

# --------------------- performance figure (Journal 1 Fig. 6) -----------
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
scatter_cols = {"train": "#2471a3", "validation": "#e67e22", "test": "#1e8449"}

for ax, (name, idx) in zip(axes.flat[:4], SPLITS):
    _, rmse, r2, _, _ = metrics(idx)
    ax.scatter(t[idx], y_all[idx], s=14, alpha=0.65,
               color=scatter_cols.get(name, "#7d3c98"))
    lo, hi = t[idx].min(), t[idx].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Y = T")
    c = np.polyfit(t[idx], y_all[idx], 1)
    ax.plot([lo, hi], np.polyval(c, [lo, hi]), "r-", lw=1.2, label="linear fit")
    ax.set_title(f"({'abcd'[SPLITS.index((name, idx))]}) {name}  "
                 f"(R$^2$={r2:.5f}, RMSE={rmse:.3f})")
    ax.set_xlabel(f"target {TARGET_TEX}")
    ax.set_ylabel(f"predicted {TARGET_TEX}")
    ax.legend(fontsize=8)

# (e) error histogram with 20 bins, stacked by split (MATLAB style)
ax = axes.flat[4]
bins = np.linspace(e_all.min(), e_all.max(), 21)
ax.hist([e_all[idx_tr], e_all[idx_va], e_all[idx_te]], bins=bins, stacked=True,
        color=[scatter_cols["train"], scatter_cols["validation"],
               scatter_cols["test"]],
        label=["training", "validation", "test"])
ax.axvline(0.0, color="#c0392b", lw=1.6, label="zero error")
ax.set_title("(e) error histogram (20 bins)")
ax.set_xlabel(f"error = target $-$ output   [{TARGET_TEX}]")
ax.set_ylabel("instances")
ax.legend(fontsize=8)

# (f) training curve
ax = axes.flat[5]
ax.semilogy(hist["iter"], hist["train"], color="#2471a3", lw=1.6, label="training MSE")
ax.semilogy(hist["iter"], hist["val"], color="#e67e22", lw=1.6, label="validation MSE")
ax.axvline(best["iter"], color="#1e8449", ls="--", lw=1.4,
           label=f"best validation (iter {best['iter']})")
ax.set_title("(f) conjugate-gradient training")
ax.set_xlabel("CG iteration")
ax.set_ylabel("MSE (normalized)")
ax.legend(fontsize=8)

fig.suptitle(f"CGNN performance for {TARGET_TEX}  -  {d} inputs, "
             f"{H} tanh hidden neurons, linear output, "
             f"70/15/15 split, {n} samples",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, FIG_PERF), dpi=160); plt.close(fig)
print(f"Saved {FIG_PERF}")

# ----------------- CGNN vs BVP along each OFAT sweep -------------------
if has_varied:
    keys = [k for k in dict.fromkeys(varied)]      # unique, in order
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow))
    xlabels = {"Br": "Br", "lam": r"$\lambda$", "G": "G", "Gr": "Gr",
               "Da": "Da", "Lam": r"$\Lambda$", "Om3": r"$\Omega_3$", "m": "m"}
    csvname = {"Br": "Br", "lam": "lambda", "G": "G", "Gr": "Gr",
               "Da": "Da", "Lam": "Lambda", "Om3": "Omega3", "m": "m"}
    for ax, key in zip(np.atleast_1d(axes).flat, keys):
        sel = np.array([v == key for v in varied])
        xcol = in_cols.index(csvname.get(key, key))
        xs = X[sel, xcol]
        style = "o" if key == "m" else "-"
        ax.plot(xs, t[sel], style, color="#2471a3", lw=2.0, ms=7,
                label="BVP (numerical)")
        ax.plot(xs, y_all[sel], "x" if key == "m" else "--", color="#c0392b",
                lw=1.8, ms=8, label="CGNN prediction")
        ax.set_xlabel(xlabels.get(key, key)); ax.set_ylabel(TARGET_TEX)
        ax.set_title(f"sweep of {xlabels.get(key, key)}")
        ax.legend(fontsize=8)
    for ax in np.atleast_1d(axes).flat[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"CGNN vs numerical solution for {TARGET_TEX} along the "
                 f"OFAT sweeps", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTDIR, FIG_SWEEP), dpi=160); plt.close(fig)
    print(f"Saved {FIG_SWEEP}")
