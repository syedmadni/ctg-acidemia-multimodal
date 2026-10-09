#!/usr/bin/env bash
# Regenerate every EJOG-revision output from the raw PhysioNet data.
# Usage: bash run_all.sh            (all steps; the deep reruns take ~2.3 h on an RTX 3050)
#        SKIP_DEEP=1 bash run_all.sh (reuse existing deep OOF in results/revision/oof/)
set -euo pipefail
cd "$(dirname "$0")"

[ -f data/raw/ctu-uhb-ctgdb/RECORDS ] || python scripts/download_data.py
# Published-pipeline arrays: needed for the published-345 cohort membership, the
# self-verification against published values, and the leak-effect comparison.
if [ ! -f data/processed/record_ids_705_anytime_vr060.csv ]; then
  python -m src.data.build_labels --config configs/baseline.yaml
  python -m src.data.build_windows --config configs/baseline.yaml --label y_acidemia_705 \
    --window-minutes 90 --window-position first --min-valid-ratio 0.60 \
    --cohort-run --cohort-name primary_705_anytime_vr060 --suffix _705_anytime_vr060
fi
python -m pytest -q tests                              # leakage, predictor and threshold tests
python -m src.revision.run_task_a                      # Task A: cohorts, LR/RF, Firth (~15 min)
if [ "${SKIP_DEEP:-0}" != "1" ]; then
  python -u -m src.revision.run_deep_revision          # Task A: deep reruns (resumable)
fi
python -m src.revision.run_task_a_deep                 # Task A: deep evaluation
python -m src.revision.run_task_a_extras               # Task A: parity 2x2, leak-effect table
python -m src.revision.figures A                       # Figs A1-A3
python -m src.revision.run_task_b                      # Task B: time to delivery, Fig. B1
python -m src.revision.run_task_c                      # Task C: operating points, Fig. C1
python -m src.revision.run_task_d                      # Task D: upper bounds, power
python -m src.revision.figures D                       # Fig. D1
python -m src.revision.run_task_f                      # Task F: manuscript Tables 1-5, Figs 1-6
echo "All revision outputs regenerated in results/revision/"
