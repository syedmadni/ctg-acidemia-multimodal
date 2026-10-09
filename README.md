# Multimodal prediction of severe fetal acidemia from CTG and maternal variables

Analysis code for the paper *"Multimodal prediction of severe fetal acidemia using
cardiotocography and maternal variables"* (revision for the European Journal of Obstetrics &
Gynecology and Reproductive Biology).

The code predicts umbilical-artery pH < 7.05 (cases) against pH ≥ 7.15 (controls) from:
- the first *t* minutes of intrapartum cardiotocography (CTG; fetal heart rate and uterine
  contractions), at checkpoints t = 10, 20, 30, 45 and 60 min;
- eight maternal variables known at admission.

It compares logistic regression and random forest on CTG summary features, CNN+transformer
networks (CTG-only and fused), and a Firth penalised likelihood-ratio test of the maternal
block. Everything a checkpoint uses comes from the signal recorded **before** that
checkpoint, and a unit test enforces this.

## Data

The CTU-UHB Intrapartum Cardiotocography Database v1.0.0 (552 recordings) is openly
available on PhysioNet. **No data are included in this repository.** Download it with:

```bash
python scripts/download_data.py        # -> data/raw/ctu-uhb-ctgdb/ (552 .hea + .dat, RECORDS)
```

Cite the database and PhysioNet as required at
<https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/>.

## Environment

Python 3.11. The exact versions are pinned in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional GPU build used for the deep models:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pytest -q tests                               # 34 tests, ~10 s once data are present
```

Deep models were trained on one NVIDIA RTX 3050 (4 GB). Classical models and all
statistics run on a CPU.

## Reproducing the results

Everything at once (about 2.5 h, mostly the deep reruns):

```bash
bash run_all.sh                 # or: SKIP_DEEP=1 bash run_all.sh to reuse saved deep predictions
```

One command per output. All outputs go to `results/revision/`, and figures are written as
300-dpi PNG and PDF files to `results/revision/figures/`.

| Output | Command |
|---|---|
| Cohort flow and sizes (`A_cohort_flow.csv`, `A_cohort_summary.csv`), LR/RF discrimination (`A_discrimination.csv`), combined vs CTG-only (`A_combined_vs_ctg.csv`), Firth LRTs (`A_firth_lrt.csv`), self-verification | `python -m src.revision.run_task_a` |
| Deep reruns (reduced encoder, 5 seeds; resumable) | `python -u -m src.revision.run_deep_revision` |
| Deep discrimination (`A_deep_*.csv`) | `python -m src.revision.run_task_a_deep` |
| Parity 2×2 and fragility, leak-effect table (`A_parity_2x2_fragility.csv`, `A_leak_effect_345.csv`) | `python -m src.revision.run_task_a_extras` |
| Figs A1-A3 | `python -m src.revision.figures A` |
| Time to delivery, second stage at t (`B_*.csv`), Fig. B1 | `python -m src.revision.run_task_b` |
| Operating points at FPR 10 %/20 % (`C_*.csv`), Fig. C1 | `python -m src.revision.run_task_c` |
| AUC-gain upper bounds, power simulation (`D_*.csv`) | `python -m src.revision.run_task_d` |
| Fig. D1 | `python -m src.revision.figures D` |
| Manuscript Tables 1-5 (`tables_final/`) and Figs 1-6 (`figures/Fig*`) on the new primary cohort | `python -m src.revision.run_task_f` |

The first run reads all 552 WFDB records once and caches them in `results/revision/cache/`.

### Fixed cross-validation folds

RepeatedStratifiedKFold (5 folds × 3 repeats, seed 42), with an inner 80/20
train/validation split. The fold assignment for every cohort and checkpoint is stored in
`results/revision/splits/` (record ID, repeat, fold, role) and read from disk by every
analysis. Imputation, scaling and class weights are fitted on training rows only.


## Repository layout

```text
src/revision/      EJOG revision pipeline (leak-free features, cohorts, tasks A-D, figures)
src/analysis/      Firth regression (validated against R logistf), DeLong test, earlier analyses
src/training/      cross-validation harness and deep-model training
src/models/        CNN+transformer encoder, fusion models
src/data/          WFDB reading, labels, windowing (published pipeline)
tests/             leakage, predictor-safety and threshold tests
scripts/           data download
configs/           YAML configs for the published pipeline
```

## Published (pre-revision) pipeline

The originally submitted analysis filled signal gaps by interpolation across the whole
90-min window (so values before a checkpoint could depend on later samples) and computed the
60 % quality ratio over a zero-padded 90-min window. The revision replaces both with
checkpoint-causal versions. `run_all.sh` rebuilds the original arrays automatically, because
the revision uses them for the published-cohort comparison. To rebuild them by hand, and to
rerun the original classical analysis:

```bash
python -m src.data.build_labels --config configs/baseline.yaml
python -m src.data.build_windows --config configs/baseline.yaml --label y_acidemia_705 \
  --window-minutes 90 --window-position first --min-valid-ratio 0.60 \
  --cohort-run --cohort-name primary_705_anytime_vr060 --suffix _705_anytime_vr060
python -m src.data.audit_valid_ratio --config configs/baseline.yaml --label y_acidemia_705
python -m src.analysis.run_featfix_cohort
python -m src.analysis.run_maternal_value_steps123
```

The three published reference tables that the revision checks itself against are included
under `outputs/quality_cohorts/` (`step1_lrt_maternal_block.csv`,
`step3_bootstrap_auc_diff.csv`, `featfix_results.csv`).

`run_task_a` checks that the revision code reproduces the published maternal-block χ² and the
LR CTG+maternal AUC from these arrays to 4 decimal places before it runs anything new.

## Firth validation

`src/analysis/firth.py` was validated against R `logistf` 1.26.1 on logistf's `sex2`
example data (coefficients, standard errors and LRT to 3 decimal places). The reference
outputs are in `src/analysis/firth_ref/`. To rerun the check, install R and `logistf`, export
the data with `Rscript -e 'library(logistf); data(sex2); save(sex2, file="src/analysis/firth_ref/sex2.rda")'`,
then run `Rscript src/analysis/firth_ref/sex2_reference.R` and
`python src/analysis/firth_ref/validate_firth.py`.
logistf's data and source are GPL and are not redistributed here.

## License

Code: MIT (see `LICENSE`). Data: PhysioNet terms (not redistributed).
