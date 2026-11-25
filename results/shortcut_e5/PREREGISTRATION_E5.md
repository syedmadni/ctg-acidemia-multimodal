# Pre-registered analysis plan – experiment E5 (plain end marker, early stopping on validation loss)

25 Nov 2026 · Syed Muhammad Madni Raza Ali

This plan adds one experiment to the shortcut experiments E1–E4. The plan for E1–E4 is in
`results/shortcut/PREREGISTRATION.md` and was made public in commit 0ed3a75 on 24 Nov 2026.

- This plan was written **after** the results of E1–E4 were known.
- Before E5 is run, this plan and its code are made public on GitHub. The commit link and its time are recorded in
  `results/shortcut_e5/REGISTRATION_E5.md`.
- E1–E4, their code and their analysis are not changed.
- Every later change is logged in `results/shortcut_e5/DEVIATIONS.md` with its date and reason, and is reported in the paper.

## 1. Why this experiment

E2 tested whether a network trained on conventional 90-min inputs relies on the end of the recording. Hiding the
end raised the AUC instead of lowering it. Two features of E2 may explain this result, and E5 changes both of them.

1. **The end of the recording was marked weakly.** In F1, as in `src/data/build_windows.py`, the FHR channel keeps
   the last valid value after the end of the recording, and only UC is 0. The end therefore looks like a flat FHR line.
2. **Early stopping was unstable.** Every inner validation set held 78 recordings, of which only 6 had acidemia.
   Early stopping on validation PR-AUC chose epoch 1 in 20 of the 75 E2 folds.

**Q5:** When the end of the recording is marked plainly and early stopping uses the validation loss, does a network
trained on real signal rely on the end of the recording?

## 2. What is already known before this registration

All 487 labelled recordings (40 with acidemia), from `results/shortcut/RESULTS_SHORTCUT.md`:

- **E1 (envelope only):**
  - AUC 0.634 (95% CI 0.546–0.714), Holm p 0.007;
  - recording-length-adjusted AUC (AAUC, tertiles) 0.527.
- **E2:**
  - AUC(F1) 0.695 and AUC(F1-x) 0.731;
  - ΔAUC −0.036 (−0.059 to −0.013), which is opposite to the hypothesis.
- **E3:** ΔAUC 0.035 (−0.051 to 0.123), Holm p 0.436.
- **E4, AUC − AAUC (tertiles):**
  - F1: −0.008 (−0.034 to 0.015);
  - F2: 0.107 (0.047 to 0.169);
  - F3: 0.047 (0.016 to 0.077).
- **E2 training:** best epoch 1 in 20 of 75 folds.

## 3. Data, cohort, validation and network (unchanged from E1–E4)

- **Cohort:** all 487 labelled recordings (40 with acidemia). Sensitivity cohort: the primary cohort at 60 min
  (474 recordings, 33 with acidemia).
- **Folds:** the same fixed folds from `results/revision/splits/splits_unfiltered_t60.csv`.
- **Network:** the same reduced CNN–transformer encoder: d_model 64, 1 layer, 4 heads, dropout 0.3; CTG-only head.
- **Training:**
  - the same optimiser and settings: AdamW, learning rate 5 × 10⁻⁴, weight decay 10⁻⁴, batch size 32;
  - at most 40 epochs, patience 8;
  - binary cross-entropy with the positive class weighted by the ratio of newborns without to with acidemia in the
    inner training split;
  - no hyperparameter tuning.
- **Standardisation:** per channel, on the inner training split, over all time steps of the input, as in E1–E3.
- **Seeds and predictions:** seeds 42–46 and the same seed rule as E1–E3. Out-of-fold predictions are averaged over
  the 5 seeds and the 3 repeats.

## 4. What changes from E2 (only these two things)

| Item | E2 | E5 |
|---|---|---|
| Input after the end of the recording | F1: FHR keeps the last valid value; UC is 0 | F5: FHR is 0 and UC is 0 |
| Early-stopping criterion | Highest validation PR-AUC | Lowest validation loss: the same class-weighted binary cross-entropy as in training, with the positive-class weight from the inner training split. The weights from the epoch with the lowest validation loss are restored. |

**Input formats:**

- **F5 (training, and evaluation (a)):**
  - Identical to F1 for every second before the end of the recording.
  - Every second at and after the end is set to 0 in both channels, in raw units, before standardisation.
  - Valid FHR is > 0 and gaps inside the recording are filled by interpolation, so FHR = 0 appears only after the end.
  - For a recording of 90 min or longer, F5 equals F1.
- **F5-x (evaluation (b)):**
  - The part after the end of the recording is filled by repeating the recording's own signal from its start, by the
    same rule as F1-x.
  - Because F5 equals F1 before the end, F5-x is identical to F1-x.

## 5. Primary outcome and test

- **Design:** train on F5, then evaluate the same trained networks on the same recordings in F5 and in F5-x.
- **Primary outcome:** paired ΔAUC = AUC(F5) − AUC(F5-x), in all 487 recordings.
- **Hypothesis:** ΔAUC > 0.
- **Test:**
  - a paired two-sided DeLong test;
  - a paired stratified bootstrap 95% CI for ΔAUC, with 2000 resamples and the same bootstrap seed as E1–E4.
- **Multiplicity:**
  - One primary test, at α = 0.05.
  - E5 is a separate family registered after E1–E4. It is not added to the Holm correction of E1–E3, and the p-values
    of E1–E3 are not recomputed.

## 6. Secondary analyses (descriptive)

1. **Dependence on recording length.**
   - AUC − AAUC (Janes–Pepe) for F5 and for F5-x, with paired bootstrap 95% CIs.
   - Strata are tertiles of recording length, with the same cut-points as E4 (from all 487 recordings, without the
     outcome). Quintiles are used as a robustness check.
2. **Size of the shift:** mean |p(F5) − p(F5-x)|.
3. **Training:** median and range of the selected epoch, and the number of folds that stopped at epoch 1, shown next to
   the same figures for E2.
4. **Comparison with the E2 networks:** AUC(F5) − AUC(F1) and AUC(F5-x) − AUC(F1-x), with paired bootstrap 95% CIs.
   The F1 and F1-x predictions come from the E2 networks.
5. **Sensitivity cohort:** the primary test and items 1–2 are repeated in the primary cohort at 60 min, using
   predictions restricted to those recordings, with no retraining.
6. **No occlusion analysis**, to limit computing time.

## 7. Interpretation rules (fixed now)

- **Primary test:**
  - Supported only if p < 0.05 and ΔAUC > 0.
  - If p < 0.05 and ΔAUC < 0, the estimate is reported as lying in the opposite direction.
  - Otherwise we report "no evidence", with the estimate and its CI.
- **Unfamiliar input.** F5-x is a kind of input the networks did not see in training, so a fall in AUC could partly
  reflect an unfamiliar input. We will say that the network relies on the end of the recording only if:
  - (i) the primary hypothesis is supported, **and**
  - (ii) the 95% CI of AUC − AAUC for F5 (tertiles) lies above 0.

  If only (i) holds, we will say that the AUC fell when the end was hidden, but that this cannot be separated from the
  effect of an unfamiliar input.
- **Primary test not supported:** we will say that, even with a plainly marked end and early stopping on validation
  loss, we found no evidence that the network relied on the end of the recording.
- **Two changes at once.** E5 changes two things compared with E2, so it cannot show which of the two matters.
- **Reporting:** E5 is reported in full, whatever its result, together with E1–E4. This is the last experiment on this
  question before the paper is submitted.

## 8. Registration, outputs and reporting

- **Time stamp.** Before E5 is run, the following files are pushed to the public repository
  github.com/syedmadni/ctg-acidemia-multimodal. The commit link and its UTC time are recorded in
  `results/shortcut_e5/REGISTRATION_E5.md`.
  - `results/shortcut_e5/PREREGISTRATION_E5.md`
  - `results/shortcut_e5/.gitignore`
  - `src/shortcut/inputs_e5.py`
  - `src/shortcut/run_e5.py`
  - `src/shortcut/analyze_e5.py`
  - `tests/test_e5_inputs.py`
- **Registered files of E1–E4 stay unchanged.**
- **Outputs** go to `results/shortcut_e5/`. Record-level files are not committed.
- **Reporting:** TRIPOD+AI. Every number in the paper comes from the result files.
