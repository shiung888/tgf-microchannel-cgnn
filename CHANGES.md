# Changes from `original/` to `src/`

Nothing in `original/` was edited. Left out of it: caches, logs, one unused calibration script, and `.npy`/`.npz`/`.pkl` copies of results.

| # | Problem in the first study | Change |
|---|---|---|
| 1 | CG and Adam scripts drew different initial weights (CG drew them after the split permutation) | `network.py` gives every method the same weights and split for a seed |
| 2 | `ann_training_dataset_ofat.csv` was rounded to 3 significant figures, so the code no longer gave the reported numbers | `make_datasets.py` rewrites it at full precision (1205 rows) |
| 3 | Br, λ, G and Gr sweeps stopped at the baseline, so they could never reach runaway | `critical_points.py` follows each branch to its turning point |
| 4 | Surrogate was tested only on points between training points of the same sweeps | `train_surrogate.py` adds Latin hypercube training data and 400 independent test points |
| 5 | Batch Adam had 1500 full gradient evaluations, CG about 5500; learning rate not tuned | `compare_optimizers.py` adds a matched budget and learning rates picked on validation error |
| 6 | One seed only | 10 seeds, with spread and number of wins |
| 7 | Sweep stops after 5 failures, but skipped points were counted as failures (13 tested, 42 skipped) | counts are printed separately |
| 8 | `np.trapezoid` does not exist in NumPy 1.x | fallback added |
| 9 | Model copied into every script | one copy in `tgf_model.py` for the new scripts |
| 10 | No script for the Newtonian limit check | `verify_solver.py` |
| 11 | No list of package versions | `requirements.txt` |

`tgf_full_analysis.py` and `parametric_plots.py` are unchanged apart from where they save files; the CSV tables are identical
to the originals.

## What changed in the conclusions

- Runaway at m = 0.5 is reached first through Gr (1.1 %) and Br (1.3 %), not through Da (32 %) or Ω₃ (23 %).
- The OFAT surrogate cannot be used for general parameter exploration. The Latin hypercube one can, away from runaway.
- CG is the best method only for Nu on the OFAT data. The large advantage over batch Adam came mainly from its small budget
  and untuned learning rate.
