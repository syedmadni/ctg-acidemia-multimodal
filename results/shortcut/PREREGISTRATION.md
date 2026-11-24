# Pre-registered analysis plan – shortcut experiments

24 Nov 2026 · Syed Muhammad Madni Raza Ali

This plan is fixed before any of the experiments below is run. Before any experiment is run, it will be made public on GitHub together with the exact code, and every later change will be logged with a date and a reason. Code: [github.com/syedmadni/ctg-acidemia-multimodal](https://github.com/syedmadni/ctg-acidemia-multimodal), to which this plan and the new scripts in src/shortcut/ are pushed before any run. The commit link and its date are recorded in results/shortcut/REGISTRATION.md.

## 1. Aim and research questions

Every CTU-UHB recording ends at delivery, and the second stage starts at minute 60 in almost every recording. As a result, the recording's length and its end carry the duration of the second stage, which is known only after birth. We test whether deep models trained in common ways learn this *shortcut* instead of fetal information.

- **Q1 (E1):** Can a model that sees no fetal heart rate or uterine activity, only where each recording ends, discriminate acidemia?
- **Q2 (E2):** Does a model trained on conventional fixed-length, zero-padded inputs with real signal use the end-of-recording marker?
- **Q3 (E3):** How much does evaluation on windows anchored to delivery (the last 60 min before birth) overstate performance at the time a real-time prediction could be made?
- **Q4 (E4):** How much of each model's discrimination remains after adjusting for recording length?

## 2. What is already known before registration

These results were seen before this plan was written. We list them so that readers can judge which findings are new.

- **Recording structure:**
  - the second stage began at 60.0 min in 493 of 506 recordings with a recorded start;
  - recording length equals 60 min plus second-stage duration (Spearman ρ 0.977–0.991).
- **Recording length alone as a marker:** AUC 0.675 (95% CI 0.590–0.754) in all 487 labelled recordings, and 0.707 (0.620–0.792) in the primary cohort at 60 min.
- **Leak-free anytime models at 60 min, all 487 recordings:**
  - deep CTG-only network: AUC 0.727 (0.642–0.806);
  - fusion network: 0.716;
  - logistic regression on CTG features: 0.743.
- **Interpolation look-ahead:** on 345 recordings, interpolation across the checkpoint raised the 60-min AUC of four classical models by 0.035–0.051.
- **Earlier exploratory models (results seen):**
  - models on the last 30 min before delivery (configs/baseline.yaml);
  - deep models on the first 90 min, zero-padded and interpolated, trained with random crops in the 345-recording cohort (configs/anytime_cohort.yaml and the step7 configs).
- **Not yet run:** the specific inputs, cohort, training regimes and evaluations defined below.

## 3. Data, cohort, validation and training

- **Data:** CTU-UHB intrapartum CTG database v1.0.0 (PhysioNet), 552 recordings.
- **Outcome:** umbilical artery pH < 7.05 (acidemia) against pH ≥ 7.15. Recordings with 7.05 ≤ pH < 7.15 are excluded.
- **Cohort:** all 487 labelled recordings (40 with acidemia), with no quality filter. This avoids the earlier choice of the 0.60 threshold. Sensitivity cohort: the primary cohort at 60 min (474 recordings, 33 with acidemia).
- **Validation:** the existing fixed folds in `results/revision/splits/splits_unfiltered_t60.csv`: stratified 5-fold cross-validation repeated 3 times (seed 42), with an inner 80/20 split into training and validation sets. All preprocessing statistics are fitted on the inner training split only.
- **Network:** the existing reduced CNN–transformer encoder, unchanged: d_model 64, 1 transformer layer, 4 heads, dropout 0.3; CTG-only head.
- **Training:** unchanged from the revision.
  - AdamW, learning rate 5 × 10⁻⁴, weight decay 10⁻⁴, batch size 32.
  - At most 40 epochs, with early stopping after 8 epochs without a higher validation PR-AUC.
  - Binary cross-entropy with the positive class weighted by the ratio of newborns without to with acidemia.
  - No hyperparameter tuning in any experiment.
- **Seeds and predictions:** seeds 42–46. Out-of-fold predictions are averaged over the 5 seeds and the 3 repeats.
- **Predictors:** nothing recorded at or after birth enters any model input except where an experiment manipulates recording length or the end of the window on purpose, as stated in Sections 4–5.

## 4. Input formats

All inputs are 1 Hz with two channels (FHR, UC) and are standardised per channel on the inner training split.

| Code | Name | Window | Gaps inside the window | After the end of the recording | Used in |
| --- | --- | --- | --- | --- | --- |
| F0 | Leak-free anytime (existing) | [0, t) from recording start, up to 90 min | 1-s bins of valid samples; causal forward fill; a leading gap stays 0 | 0 | Reference only: existing predictions, no retraining |
| F1 | Conventional fixed length | First 90 min from recording start | Exactly as `src/data/build_windows.py`: invalid FHR (≤ 0 or non-finite) filled by linear interpolation in both directions across the whole window; UC non-finite values filled the same way; 4 Hz averaged to 1 Hz | Zero-padded to 90 min, as in `build_windows.py` (FHR tail holds the last valid value; UC tail is 0) | E2, E4 |
| F1-x | F1 with the end marker removed | As F1 | As F1 | The tail after the end of the recording is filled by repeating the recording's own F1 signal from its start, so every input looks like 90 min of recorded signal | E2, evaluation only |
| F2 | Envelope only | Same 90-min frame as F1 | Every recorded second replaced by independent N(0, 1) noise in both channels (noise seed = record ID) | 0 | E1 |
| F3 | Delivery-anchored | Last 60 min before the end of the recording, [L − 60, L) | Valid FHR: finite and > 0; valid UC: finite and ≠ 0 (as in the revision). Interior gaps ≤ 15 s filled by linear interpolation between the neighbouring valid samples; longer gaps, and gaps that touch either edge of the window, set to 0 | None: every recording lasts at least 60 min | E3 training; E3 evaluation (a) |
| F3-s | Start-anchored counterpart of F3 | First 60 min, [0, 60) | Same rule as F3, inside the window | None | E3 evaluation (b) |

F3 follows the most common design in our review: windows that end at delivery, with short gaps interpolated, as in Mendis et al. (2024; Sci Rep 14:12615, the published version of their 2023 Research Square preprint).

## 5. Experiments and primary outcomes

| Exp. | Train on | Evaluate on | Primary outcome | Hypothesis |
| --- | --- | --- | --- | --- |
| E1 | F2 | F2, out of fold | AUC of the envelope-only network | AUC > 0.5 |
| E2 | F1, the full 90-min window, with no random cropping | (a) F1 and (b) F1-x, using the same trained networks and recordings | Paired ΔAUC = AUC(F1) − AUC(F1-x) | ΔAUC > 0 |
| E3 | F3 | (a) F3 and (b) F3-s, using the same trained networks and recordings | Paired ΔAUC = AUC(F3) − AUC(F3-s) | ΔAUC > 0 |
| E4 | No new training | Predictions from E1, E2(a) and E3(a), plus the existing F0 deep CTG-only network and leak-free LR CTG-only, at 60 min | Recording-length-adjusted AUC (AAUC) of each model, and AUC − AAUC | Estimation only; no test |

**What each experiment can show:**

- **E1:** if the network reaches AUC > 0.5 with no fetal signal, the input format alone carries outcome information.
- **E2:** if removing only the end marker lowers the AUC of a network trained on real signal, that network relies on the marker.
- **E3:** the gap shows how much delivery-anchored evaluation overstates what the same network achieves when only the signal available in real time at 60 min is used. It does not separate a shortcut from the true value of the final minutes before birth, and we will report it with that caveat.
- **E4:** the share of each model's discrimination that does not depend on recording length.

## 6. Statistical analysis and interpretation rules

- **AUC and confidence intervals:** AUC is computed on the out-of-fold predictions averaged over seeds and repeats. 95% CIs come from 2000 bootstrap resamples of recordings, stratified by outcome (percentile method).
- **E1 test:** a two-sided z-test of AUC = 0.5, using the DeLong variance.
- **E2 and E3 tests:** a paired two-sided DeLong test of the two AUCs, plus a paired stratified bootstrap 95% CI for ΔAUC.
- **Multiplicity:** Holm correction across the three primary tests (E1–E3), at α = 0.05.
- **E4 (Janes and Pepe covariate-adjusted AUC):**
  - Recordings are grouped into tertiles of recording length, with cut-points taken from all 487 recordings without using the outcome.
  - AAUC = Σ (n₁ₛ n₀ₛ AUCₛ) / Σ (n₁ₛ n₀ₛ). This is the probability that a newborn with acidemia scores higher than a newborn without acidemia whose recording is of similar length.
  - 95% CIs by stratified bootstrap. AUC − AAUC is reported with a paired bootstrap CI.
- **Interpretation rules, fixed now:**
  - A primary hypothesis is supported only if its Holm-adjusted p < 0.05 and the estimate lies in the stated direction.
  - Otherwise we report "no evidence", together with the estimate and its CI.
  - Every experiment is reported, whatever its result. None will be dropped or replaced.
  - We will not describe a shortcut as "established" on the basis of E3 alone, because E3 cannot separate a shortcut from the true value of signal recorded near birth.

## 7. Secondary and descriptive analyses

- **Sensitivity cohort:** E1–E4 are repeated in the primary cohort at 60 min (474 recordings, 33 with acidemia), using predictions restricted to those recordings, with no retraining.
- **Size of the shift in E2:** the mean absolute change in predicted risk between F1 and F1-x.
- **Occlusion profile in E2:** for networks trained on F1 with seed 42 (all 15 folds; one seed only, to limit computing time), each 2-min block is replaced in turn by that recording's mean, and the mean absolute change in the logit is recorded. The profile is plotted against time relative to the end of the recording, from −60 to +30 min. Descriptive only.
- **Real-time use of F1 networks:** networks trained on F1 are evaluated on F1 inputs cut at 60 min, with everything after 60 min set to padding as if the recording ended there. This AUC is compared descriptively with the leak-free F0 network at 60 min.
- **E4 robustness:** E4 is repeated with recording-length quintiles instead of tertiles. AAUC is also reported, as descriptive extras, for F1-x, F1c60, F3-s and recording length alone.
- **Reference value:** the AUC of recording length alone in the same cohort, for comparison with E1.

## 8. Deviations, reporting and registration

- **Time stamp:** before any experiment is run, this plan (as results/shortcut/PREREGISTRATION.md) and the exact code are pushed to the public repository github.com/syedmadni/ctg-acidemia-multimodal. The commit link and its date are recorded in results/shortcut/REGISTRATION.md and cited in the paper.
- **Deviations:** any change after registration is logged in `results/shortcut/DEVIATIONS.md` with its date and reason, and is reported in the paper.
- **Data and code:** all results, predictions and figures go to `results/shortcut/`. The code commit used for the runs is named in the paper.
- **Reporting:** TRIPOD+AI. Every number in the paper comes from the result files.
