"""
Gradient-descent optimizer comparison for the ANN surrogate, all using the
Adam optimizer:

    * Batch GD     - one Adam update per epoch (batch = full training set)
    * Mini-batch   - Adam updates on batches of 32 samples
    * SGD          - Adam updates on single samples (batch = 1)

Same network as the CGNN scripts (1 hidden layer, 10 tanh neurons, linear
output), same 70/15/15 split and mapminmax [-1,1] normalization, so results
are directly comparable to the conjugate-gradient (CG) baseline.

Data   : ../ann_training_dataset_ofat.csv   (8 inputs: m, Br, lambda, G, Gr,
         Da, Lambda, Omega3), targets Nu_wall (Eq. 23) and Sxy_wall (Eq. 22).
Adam   : lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 (same for all three
         methods, so the only difference is the batch size / gradient noise).

Usage:
    python gd_adam_comparison.py            # runs everything (both targets + plots)
    python gd_adam_comparison.py Nu_wall    # compute one target, cache to .pkl
    python gd_adam_comparison.py plot        # build figures/CSV from caches

Outputs (in this folder):
    fig_convergence.png   training/validation MSE vs epoch, 3 methods x 2 targets
    fig_regression.png    predicted vs target scatter, 3 methods x 2 targets
    fig_metrics.png       test R^2 and RMSE bar charts per method
    gd_adam_results.csv    MSE/RMSE/R^2 per method, target and split + timing
"""

import os
import sys
import csv
import time
import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(os.path.dirname(HERE), "ann_training_dataset_ofat.csv")

# ------------------------------ settings -------------------------------
N_HIDDEN = 10
EPOCHS   = 1500
LR       = 1e-3
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
SEED     = 7
TARGETS  = ["Nu_wall", "Sxy_wall"]
METHODS  = [("Batch GD", None), ("Mini-batch (32)", 32), ("SGD (1)", 1)]
COLORS   = {"Batch GD": "#2471a3", "Mini-batch (32)": "#e67e22", "SGD (1)": "#1e8449"}

# ------------------------------ load data ------------------------------
with open(CSV_FILE) as f:
    rdr = list(csv.reader(f))
header, data = rdr[0], rdr[1:]
in_cols = [c for c in header if c not in ("varied", "Sxy_wall", "Nu_wall")]
in_idx = [header.index(c) for c in in_cols]
X = np.array([[float(r[i]) for i in in_idx] for r in data])
n, d = X.shape

rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
n_tr, n_va = int(0.70 * n), int(0.15 * n)
idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
SPLITS = [("train", idx_tr), ("validation", idx_va),
          ("test", idx_te), ("all", np.arange(n))]

Xmin, Xmax = X[idx_tr].min(0), X[idx_tr].max(0)
Xn = 2 * (X - Xmin) / (Xmax - Xmin) - 1

# --------------------------- network helpers ---------------------------
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
    hid = np.tanh(Xin @ W1 + b1)
    return hid @ W2 + b2


def loss_grad(w, Xin, tgt):
    W1, b1, W2, b2 = unpack(w)
    hid = np.tanh(Xin @ W1 + b1)
    out = hid @ W2 + b2
    err = out - tgt
    mb = Xin.shape[0]
    d_out = 2.0 * err / mb
    gW2 = hid.T @ d_out
    gb2 = d_out.sum()
    d_hid = np.outer(d_out, W2) * (1 - hid ** 2)
    gW1 = Xin.T @ d_hid
    gb1 = d_hid.sum(0)
    return np.concatenate([gW1.ravel(), gb1, gW2, [gb2]])


def mse(w, Xin, tgt):
    return float(np.mean((forward(w, Xin) - tgt) ** 2))


def train_adam(w0, Xtr, ttr, Xva, tva, batch, shuffle_seed):
    w = w0.copy()
    m = np.zeros_like(w); v = np.zeros_like(w)
    step = 0
    ntr = Xtr.shape[0]
    bs = ntr if batch is None else batch
    rsh = np.random.default_rng(shuffle_seed)
    hist_tr, hist_va = [], []
    best_w = w.copy(); best_va = np.inf; best_ep = 0     # best-validation snapshot
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rsh.permutation(ntr)
        for start in range(0, ntr, bs):
            b = order[start:start + bs]
            g = loss_grad(w, Xtr[b], ttr[b])
            step += 1
            m = BETA1 * m + (1 - BETA1) * g
            v = BETA2 * v + (1 - BETA2) * g * g
            mhat = m / (1 - BETA1 ** step)
            vhat = v / (1 - BETA2 ** step)
            w = w - LR * mhat / (np.sqrt(vhat) + EPS)
        mtr = mse(w, Xtr, ttr); mva = mse(w, Xva, tva)
        hist_tr.append(mtr); hist_va.append(mva)
        if mva < best_va:                                # keep lowest-val weights
            best_va = mva; best_w = w.copy(); best_ep = ep
    # return the best-validation weights, not the final-epoch weights
    return best_w, np.array(hist_tr), np.array(hist_va), time.time() - t0, step, best_ep


w0 = np.random.default_rng(SEED).normal(0.0, 0.5, n_w)   # shared init


def compute_target(target):
    t_idx = header.index(target)
    t = np.array([float(r[t_idx]) for r in data])
    tmin, tmax = t[idx_tr].min(), t[idx_tr].max()
    tn = 2 * (t - tmin) / (tmax - tmin) - 1
    denorm = lambda yn: (yn + 1) / 2 * (tmax - tmin) + tmin
    out = {}
    print(f"\n=== target {target} ===")
    print(f"{'method':<18}{'time(s)':>9}{'updates':>10}{'best_ep':>9}"
          f"{'R2_train':>10}{'R2_val':>9}{'R2_test':>9}{'RMSE_test':>11}")
    for name, batch in METHODS:
        w, htr, hva, secs, nstep, best_ep = train_adam(
            w0, Xn[idx_tr], tn[idx_tr], Xn[idx_va], tn[idx_va],
            batch, shuffle_seed=SEED + 1)
        y_all = denorm(forward(w, Xn))
        e_all = t - y_all
        rec = {"time": secs, "updates": nstep, "best_ep": best_ep,
               "htr": htr, "hva": hva, "y_all": y_all, "t": t}
        for sname, sidx in SPLITS:
            err = e_all[sidx]
            mse_s = float(np.mean(err ** 2))
            rec[sname] = (mse_s, float(np.sqrt(mse_s)),
                          float(1 - np.sum(err ** 2) /
                                np.sum((t[sidx] - t[sidx].mean()) ** 2)))
        out[name] = rec
        print(f"{name:<18}{secs:>9.2f}{nstep:>10}{best_ep:>9}"
              f"{rec['train'][2]:>10.5f}{rec['validation'][2]:>9.5f}"
              f"{rec['test'][2]:>9.5f}{rec['test'][1]:>11.4f}")
    with open(os.path.join(HERE, f"_cache_{target}.pkl"), "wb") as fp:
        pickle.dump(out, fp)
    return out


def make_plots():
    res = {}
    for target in TARGETS:
        with open(os.path.join(HERE, f"_cache_{target}.pkl"), "rb") as fp:
            res[target] = pickle.load(fp)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    mnames = [m[0] for m in METHODS]

    # --- fig 1: convergence ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target in zip(axes, TARGETS):
        for name in mnames:
            rec = res[target][name]
            ax.semilogy(rec["htr"], color=COLORS[name], lw=1.8, label=f"{name} (train)")
            ax.semilogy(rec["hva"], color=COLORS[name], lw=1.0, ls="--", alpha=0.7,
                        label=f"{name} (val)")
            be = rec["best_ep"]
            ax.plot(be, rec["hva"][be], "o", color=COLORS[name], ms=8, mec="k",
                    mew=1.0, zorder=5,
                    label=f"{name} best-val (ep {be})")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalized)")
        ax.set_title(f"convergence - {target}"); ax.legend(fontsize=7.5)
    fig.suptitle(f"Adam training convergence: Batch GD vs Mini-batch vs SGD "
                 f"(lr={LR}, {EPOCHS} epochs)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, "fig_convergence.png"), dpi=160); plt.close(fig)
    print("Saved fig_convergence.png")

    # --- fig 2: regression ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for row, target in enumerate(TARGETS):
        for col, name in enumerate(mnames):
            ax = axes[row, col]; rec = res[target][name]
            t = rec["t"]; y = rec["y_all"]
            ax.scatter(t, y, s=12, alpha=0.5, color=COLORS[name])
            lo, hi = t.min(), t.max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Y = T")
            c = np.polyfit(t, y, 1)
            ax.plot([lo, hi], np.polyval(c, [lo, hi]), "r-", lw=1.2, label="fit")
            ax.set_title(f"{name} - {target}\n"
                         f"(R$^2$={rec['all'][2]:.5f}, RMSE={rec['all'][1]:.3f})")
            ax.set_xlabel("target"); ax.set_ylabel("predicted"); ax.legend(fontsize=8)
    fig.suptitle("Predicted vs target (all data) - Adam optimizer, 3 batch regimes",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(HERE, "fig_regression.png"), dpi=160); plt.close(fig)
    print("Saved fig_regression.png")

    # --- fig 3: metrics bars ---
    xpos = np.arange(len(mnames)); w_bar = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for j, target in enumerate(TARGETS):
        r2s = [res[target][m]["test"][2] for m in mnames]
        rmses = [res[target][m]["test"][1] for m in mnames]
        axes[0].bar(xpos + (j - 0.5) * w_bar, r2s, w_bar, label=target)
        axes[1].bar(xpos + (j - 0.5) * w_bar, rmses, w_bar, label=target)
    lo = min(res[t][m]["test"][2] for t in TARGETS for m in mnames)
    axes[0].set_ylabel("test R$^2$"); axes[0].set_title("(a) test R$^2$ by method")
    axes[0].set_ylim(min(0.0, lo) - 0.05, 1.0)
    axes[1].set_ylabel("test RMSE"); axes[1].set_title("(b) test RMSE by method")
    for ax in axes:
        ax.set_xticks(xpos); ax.set_xticklabels(mnames); ax.legend()
    fig.suptitle("Test-set accuracy by batch regime (Adam)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, "fig_metrics.png"), dpi=160); plt.close(fig)
    print("Saved fig_metrics.png")

    # --- CSV ---
    with open(os.path.join(HERE, "gd_adam_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "method", "batch_size", "epochs", "updates",
                    "best_val_epoch", "train_time_s", "split", "MSE", "RMSE", "R2"])
        for target in TARGETS:
            for name, batch in METHODS:
                rec = res[target][name]
                bs = n_tr if batch is None else batch
                for sname, _ in SPLITS:
                    m_, rm_, r2_ = rec[sname]
                    w.writerow([target, name, bs, EPOCHS, rec["updates"],
                                rec["best_ep"], f"{rec['time']:.2f}", sname,
                                f"{m_:.6e}", f"{rm_:.6f}", f"{r2_:.6f}"])
    print("Saved gd_adam_results.csv")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "plot":
        make_plots()
    elif arg in TARGETS:
        compute_target(arg)
    else:
        for tg in TARGETS:
            compute_target(tg)
        make_plots()
