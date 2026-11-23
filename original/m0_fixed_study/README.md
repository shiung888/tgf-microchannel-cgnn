# Fixed-m study (m = 0, Arrhenius only)

Variant of the main pipeline with the reaction-kinetics index **m fixed at 0**
instead of crossed over m = -2, 0, 0.5. Your original files are unchanged.

## What was changed vs the originals
- `ann_dataset_ofat_m0.py`  : `M_VALUES = [0.0]` (was `[-2.0, 0.0, 0.5]`).
- `tgf_full_analysis_m0.py` : `M_VALUES = [0.0]`.
- `cgnn_nu_m0.py`, `cgnn_sxy_m0.py` : the constant `m` column is **dropped**
  from the ANN inputs. Required — with a single m value the m column has zero
  range, so the original mapminmax normalization divides by zero and every
  weight becomes NaN (training stops at iteration 0). Inputs are now the 7
  genuinely-varying parameters: Br, lambda, G, Gr, Da, Lambda, Omega3.

## Results (this m = 0 dataset)
- OFAT dataset: 420 samples, 0 discarded (m = 0 never hits thermal runaway).
- CGNN Nu(y=1) : R2(all) = 0.99967, RMSE = 0.036
- CGNN Sxy(y=1): R2(all) = 0.99955, RMSE = 0.053

## Compared to the crossed-m baseline (m = -2, 0, 0.5, 1205 samples)
- Baseline Nu : R2 = 0.9907, RMSE = 0.261
- Baseline Sxy: R2 = 0.9945, RMSE = 0.222

The higher R2 here is NOT a better model — it is an easier, narrower problem
(one input dimension removed, only the well-behaved m = 0 case, fewer samples).

## Disadvantages of fixing m = 0
1. Breaks the ANN script as-is (constant m column -> NaN); m must be dropped.
2. The model cannot predict at any other m; the strong m-sensitivity of Nu
   (the reason the crossed-m dataset exists) is lost entirely.
3. Fewer training samples (420 vs 1205) -> less general coverage.
4. No reaction-kinetics comparison (sensitized / Arrhenius / bimolecular),
   which is a core physical contribution of the study.

Only genuine upside: cleaner dataset (every sweep point converges at m = 0).
