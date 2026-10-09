# EJOG revision: deviations and frozen analysis choices

**Frozen 2026-09-28, before any revision result was computed.** Everything in sections 1-3
was fixed after the Step 0 audit (code reading plus signal-level counts that use no outcome
labels) and before any model, test or cohort outcome count was run. Later changes go in
section 4 with a date and a reason. Nothing in sections 1-3 is edited after results exist.

Plan of record: the author-approved revision plan (approved with amendments, 2026-09-28;
kept locally, not in this repository).

## 1. Step 0 audit findings that motivate the revision

| # | Finding | Evidence |
|---|---|---|
| F1 | The published quality ratio is **not** over the whole recording. It is `valid.mean()` over a fixed 90-min (21,600-sample) first window, **zero-padded at the tail**, so padding counts as invalid and short recordings are penalised. 121/552 records have whole-recording ratio ≥ 0.60 but padded ratio < 0.60 (corr. between the two ratios 0.79). The filter is partly a recording-length filter. | `src/data/build_windows.py:150-154`, `:47-57` |
| F2 | **Interpolation look-ahead.** Invalid FHR is filled by linear interpolation with `limit_direction="both"` over the whole 90-min window before cropping. For 90/63/70/84/88 of 552 records at 10/20/30/45/60 min, values inside [0,t) depend on samples after t. 39 records have no valid FHR in [0,10) and are back-filled from later data. The published CTG features and the deep-model inputs both use these arrays, so features described as "valid samples only" were computed over interpolated signal. | `build_windows.py:15-23`, `:171`; `src/data/fixed_ctg_features.py:66-67` |
| F3 | `Sig2Birth` is present in every header and **equals 0 in all 552 records**. `Pos. II.st.` is set for 506 and is −1 for 46. | all `data/raw/ctu-uhb-ctgdb/*.hea` |
| F4 | Minimum recording length is 60.0 min (median 71.7, max 90.08), so "length ≥ t" excludes nobody at 10-60 min. | raw signal lengths |
| F5 | Published bootstrap CIs used 1000 **unstratified** resamples (`train_cv.py:1111-1116`; `run_quality_threshold_sensitivity.py:55,88`). The Step 3 ΔAUC used 2000 stratified (`run_maternal_value_steps123.py:62,433-451`). | as cited |
| F6 | Deep models kept only the best fold's checkpoint; validation and training predictions were never saved. `fold_metrics.csv` `threshold_at_fpr_*` are computed on the test fold (optimistic) and are not reused. | `train_cv.py:1596` |

## 2. Amendments approved by the author (frozen 2026-09-28)

A1. **New primary analysis.** Because F1 and F2 affect the published primary analysis, the old
primary results are not reported as main results. The corrected **PRIMARY** analysis is the
`ckfilter` cohort: at checkpoint t, include a record if q(t) ≥ 0.60 on FHR samples in [0,t),
the gapped label is valid, and length ≥ t. It uses causal valid-only CTG features (§3). The
**primary test** is the Firth penalised LRT for the 8-variable maternal block (8 df) on
`ckfilter`, with Holm across the 5 checkpoints (10, 20, 30, 45, 60). Sensitivity analyses:
`unfiltered` (label valid, length ≥ t, no quality filter) and `primary345_causal` (the
published 345 records with causal features). Parity (1 df) stays **POST HOC**. Old primary
results appear only in the before/after table in `CHANGES_VS_MANUSCRIPT.md`.

A2. **Deep reruns: reduced encoder only.** `ctg_only` and concat `fusion`, **5 seeds**, causal
inputs (§3), trained once per seed on `unfiltered`, evaluated at each checkpoint on the
`ckfilter`, `unfiltered` and `primary345_causal` subsets of the OOF. Validation-split
predictions are saved so Task C uses validation-control thresholds for deep models too.
**Deviation from the project rule "a deep conclusion is stated only if it holds for both
encoder sizes":** the full-size encoder is not rerun now (estimated ~25 GPU-hours for 5
seeds). Its cost is re-reported after Task A for an author decision. Deep `gated_fusion` and
deep `maternal_only` are not rerun and will leave the main text.

A3. **Extra pre-declared analyses.**
- (a) Task B: from `Pos. II.st.`, per checkpoint, n and % of recordings already in the second
  stage at t (Pos. II.st. ≤ t·240 samples), cases vs controls. −1 is reported as "unknown".
  Descriptive only.
- (b) Task A: one extra Firth LRT with an **extended intrapartum block** = the 8 maternal
  variables + Pyrexia, Meconium, Presentation, Liq. praecox. Coding and missingness are
  checked and reported first; a multi-category variable is dummy-coded and the df reported.
  Labelled "pre-declared sensitivity analysis". **Post-birth fields are never predictors:**
  pH, BDecf, BE, pCO2, Apgar1, Apgar5, NICU days, Seizures, HIE, Intubation, Main diag.,
  Other diag., Deliv. type, CK/KP, NoProgress, I.stage, II.stage, Weight(g), Sex. (Sex is
  used only descriptively in Table 1.) A unit test enforces this.
- (c) Table 1 builder: newborn Sex, cases vs controls, Fisher exact test.

A4. **Task F** (after A-E): regenerate every manuscript table and figure (Tables 1-5,
Figs 1-6) on the new primary `ckfilter` cohort with causal features. Table 5 calibration uses
out-of-fold predictions and reports both the class-weighted LR (current) and an unweighted LR.
Figures are saved separately at 300 dpi as PNG and PDF. `paper/` is not edited.

A5. **Split manifests may be committed** (record_id, repeat, fold, inner split; no clinical
data). This is an explicit, author-approved exception to the project rule "never whitelist
record-level files". All other record-level files (OOF, validation predictions, raw-signal
cache, model checkpoints) stay git-ignored.

A6. Dead code is deleted only after the author approves the final list.

## 3. Frozen analysis choices

- **Label:** gapped primary label, pH < 7.05 = case, pH ≥ 7.15 = control; 7.05 ≤ pH < 7.15
  excluded (`build_labels.py:112-114`).
- **Valid FHR sample:** finite and > 0, **unchanged** from `build_windows.py:153`. Valid UC
  sample: finite and ≠ 0 (matches the zero-as-missing convention of `ctg_summary_features_v2`).
- **quality_ratio(fhr_4hz, t)** = mean of the valid-FHR indicator over raw samples
  [0, t·60·4). Denominator t·240, no padding. Threshold 0.60 (unchanged).
- **Causal CTG features.** Same 10 statistics, same order (`fhr_mean, uc_mean, fhr_std,
  uc_std, fhr_min, uc_min, fhr_max, uc_max, fhr_median, uc_median`). From raw 4 Hz samples in
  [0,t): 1-s bins take the mean of the valid samples in the bin (NaN if none); statistics
  are taken over non-NaN bins (population SD, ddof=0, as before); an all-NaN channel gives NaN,
  imputed within fold. No interpolation. **This deviates from the published feature
  definition** (which used interpolated 1 Hz arrays; F2). It matches the manuscript's stated
  "valid samples only".
- **Causal deep inputs.** 1 Hz valid-only bins as above, then **causal forward-fill** (last
  valid value carried forward; a leading gap stays 0, the existing missing code), zero past
  the end of the recording. A crop at t therefore depends only on [0,t). Per-fold channel
  normalisation, as before. Training cutoffs are drawn from [8, min(90, duration)]
  (`duration_aware_cutoffs: true`, as in step 7d). This deviates from the published deep
  preprocessing (F2).
- **Folds:** RepeatedStratifiedKFold 5×3, seed 42, plus the inner 80/20 train/validation split
  exactly as `precompute_quality_splits` (`train_cv.py:632`), regenerated per cohort and
  checkpoint (the `ckfilter` cohort changes with t), saved to `results/revision/splits/`, and
  loaded from disk by every analysis. LR/RF are fit on inner-train only, as now.
- **Bootstrap:** **2000** patient-level resamples, **stratified by class**, seed 42, on
  repeat-averaged OOF, percentile intervals. **Deviation from published CIs** (1000,
  unstratified; F5).
- **Firth:** existing validated `src/analysis/firth.py::firth_lrt`, unchanged; whole-cohort
  median imputation, constant CTG columns dropped (as `run_maternal_value_steps123.py`).
  Holm across the 5 checkpoints within each (cohort variant × test) family.
- **Combined vs CTG-only:** paired DeLong (`src/analysis/significance.py`) plus a paired
  stratified bootstrap of ΔAUC; Holm across 5 checkpoints.
- **Task B:** time_to_delivery(t) = (length − t) + Sig2Birth; Sig2Birth = 0 for every record
  (F3), so ttd(t) = length − t. **Known before any computation:** within a fixed cohort,
  ttd(t) is length shifted by a constant, so its AUC equals that of recording length at every
  checkpoint, and adjusting for ttd(t) is equivalent to adjusting for length. It varies across
  t only through the `ckfilter` cohort changing. Explanatory confounding check only, never a
  deployable predictor.
- **Task C:** target FPR 10% and 20%. Leak-free primary: threshold = (1 − FPR) quantile
  (numpy `quantile`, linear) of the **validation-split controls'** scores in that outer fold,
  applied to the outer-test fold. **Deviation from the prompt's "training controls":**
  in-sample RF scores are near-perfect, which would make such thresholds far too low.
  Repeats: one binary call per record per repeat; metrics per repeat, then averaged over 3
  repeats, with the per-repeat range reported. Bootstrap: resample patients (stratified),
  recompute per-repeat metrics, average. Secondary (labelled optimistic): threshold chosen on
  pooled repeat-averaged OOF.
- **Task D power simulation.** Covariate rows fixed at the observed cohort (median-imputed).
  True logit = α + β_CTG·CTG + γ·s, where β_CTG is the Firth M0 fit, s is the maternal linear
  predictor in the direction of the observed Firth full-model maternal coefficients (SD 1),
  and α gives expected prevalence equal to the observed prevalence. γ is calibrated so the
  population AUC gain of the true M1 predictor over the CTG-only part equals 0.03 / 0.05 /
  0.08 / 0.10. Then 500 simulated outcome vectors per effect size, each tested with the Firth
  8-df LRT. Power is reported at α = 0.05 and α = 0.01 (strictest Holm step over 5), with the
  Monte Carlo SE, on the primary `ckfilter` cohort at every checkpoint.
- **Seeds:** global seed 42 for all resampling. Deep training seeds 42, 43, 44, 45, 46 (folds
  fixed at seed 42).

## 4. Deviations logged during the work
(dated entries appended below)

- **2026-09-28, header parser.** `build_labels.parse_hea_to_kv` misparses `#Liq. praecox 1`
  (single space), so `liq_praecox` is NaN for all 552 records in
  `data/processed/record_meta_labels_acidemia.csv`. The revision uses its own parser
  (`src/revision/raw_cache.py`, value = last token, explicit field map). It is cross-checked
  with 0 mismatches against the old file on every other field used (pH, the 8 maternal
  variables, pyrexia, meconium, presentation, sex, Apgar, BE, weight, stage lengths).
  Recovered: liq_praecox 1 = 147/552. The published analysis never used this field, so no
  published number changes. Found before any revision result was computed.
- **2026-09-28, extended block coding (A3b), fixed before the Firth run.** Presentation codes
  1/2/3 are dummy-coded with reference code 1; 3 missing values are set to code 1 (the mode).
  Pyrexia, meconium and liq. praecox are 0/1 with no missing values. Block df = 8 + 1 + 1 + 2
  + 1 = 13 (lower if a column is constant within a cohort; reported per row).
- **2026-09-28, Sex.** Added to the forbidden-predictor list (descriptive use in Table 1
  only), a conservative reading of A3b.
- **2026-09-28, deep thresholds for Task C, fixed before Task C.** Deep models are trained on
  `unfiltered`. When a deep model is evaluated on a subset (ckfilter at t, primary345), the
  validation controls used for its threshold are restricted to the same subset's records
  within that fold's validation split.
- **2026-09-28, self-verification.** Using the published v2 features, published arrays and
  splits read from disk, the pipeline reproduces the published Firth maternal chi2 and LR
  CTG+maternal AUC at all 5 checkpoints to 4 dp (`A_self_verification.csv`).
- **2026-09-29, RF row-order sensitivity (found while building the leak-effect table).**
  `load_splits` returns index arrays in sorted order; the published pipeline passed
  `train_test_split`'s shuffled order. RandomForest (bootstrap sampling) depends on
  training-row order, so identical data, seed and folds give different RF AUCs (0.7212
  sorted vs 0.7264 shuffled; up to 0.016 across the 20 leak-table cells). LR is
  order-invariant and matches the published values exactly. Task A was already run with
  sorted order and is **not** rerun (that would be a post-result change). Every revision
  comparison uses the same order in both arms. The Task A self-verification covered LR
  only, which is why it did not catch this.
- **2026-09-29, author decisions.** (1) The full-size deep encoder is **cancelled**. Deep
  results rest on the reduced encoder only (5 seeds); the project rule "both encoder sizes"
  is not met, and every deep statement is labelled "reduced encoder only". (2) Split files
  keep the sorted order; RF is not rerun (see the RF row-order entry above). (3) Scope
  freeze: B, C, D, E, F proceed in order. Real errors stop the work and are reported;
  improvements go to `FUTURE_WORK.md` only.
- **2026-09-29, Task C implementation details (fixed before viewing Task C results).** A
  record is called positive if score > threshold (strict). Deep test and validation scores are
  averaged over the 5 seeds within each (repeat, fold) before thresholding. The optimistic
  secondary uses repeat-averaged pooled OOF scores (the same scores as the AUC analysis).
- **2026-09-29, Task D additions.** A gain-0 cell (500 simulations per checkpoint) was added
  as a simulation size check; it is a validity check of the simulation, not a new analysis.
  gamma was calibrated with Brent's method on [1e-6, 10]. The population AUC counts all
  ordered pairs i != j weighted by p_i(1 - p_j), with ties counted as 0.5. It was verified
  against a brute-force computation.
- **2026-09-29, Task F mapping decisions.** (1) Table 1 uses the primary cohort at the lead
  checkpoint (60 min, 474/33), because the primary cohort changes with t; the footnote gives
  the range. (2) The Table 5 / Fig. 6 sensitivity cohort is `unfiltered` (a pre-declared
  revision sensitivity cohort), replacing the consecutive-label n = 400 cohort, which the
  revision does not re-estimate. (3) Fig. 2 has one panel, since the full-size encoder is
  cancelled. (4) Table 4 and Fig. 5 (gated fusion) are marked REMOVED rather than rebuilt.
  (5) Table 5 uses the same repeat-averaged OOF and calibration definitions as the published
  7f analysis (GLM intercept with offset, GLM slope, Wilson bins, LOWESS it=0); the unweighted
  LR is the same pipeline without `class_weight`.
- **2026-09-29, newborn sex coding (Table 1).** Checked the PhysioNet CTU-UHB page and
  Chudacek et al. 2014 (BMC Pregnancy Childbirth 14:16; full text searched). The paper states
  only "Sex of the fetus - both sexes were included"; neither source defines codes 1 and 2.
  Table 1 therefore reports "code 1 / code 2" without a male/female label.
