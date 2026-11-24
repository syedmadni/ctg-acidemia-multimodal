# Shortcut experiments (pre-registered)

Code for the four pre-registered experiments: E1 envelope-only model, E2 end-marker removal,
E3 delivery-anchored versus start-anchored evaluation, and E4 recording-length-adjusted AUC.
The analysis plan (`results/shortcut/PREREGISTRATION.md`) and this code were made public on GitHub
before any of these runs; the commit link and its date are in `results/shortcut/REGISTRATION.md`.
E2 occlusion is computed for seed 42 only, to limit computing time.

## Run (GPU recommended; several hours on an RTX 3050)

```bash
python -m pytest -q tests/test_shortcut_inputs.py      # 6 tests, no data needed
python -m src.shortcut.run_shortcut                     # E1-E3: 3 x 15 folds x 5 seeds (resumable)
python -m src.shortcut.analyze                          # statistics and E4 -> results/shortcut/analysis/
```

`run_shortcut` needs the CTU-UHB data (`python scripts/download_data.py`) and the fold files in
`results/revision/splits/`. If `results/revision/oof/` exists from the revision run, the
leak-free reference predictions are copied for E4; otherwise E4 reports the new models only.

A quick pipeline check that runs on a CPU in a few minutes (not for results):

```bash
python -m src.shortcut.run_shortcut --smoke
```
