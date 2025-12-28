# Build Log - ScratchML (CPU-only)

## Goal
Implement and validate binary logistic regression from scratch (PyTorch core, manual optimization), compare Gradient Descent vs Newton, and verify against a PyTorch baseline. Produce reproducible artifacts (results + plots).

## Timeline (from file timestamps)

### 12/27/2025
- 5:30 PM - `build_log.md` created (initial log placeholder).
- 5:33 PM - `scratchml/decomposition.py` created/updated.
- 5:47 PM - `scratchml/clustering.py` created/updated.
- 6:01 PM - Virtual environment folder `.venv/` created.
- 6:25 PM - `requirements.txt` created/updated.

### 12/28/2025
- 10:42 AM - Core utilities/package bootstrap updated:
  - `scratchml/__init__.py`
  - `scratchml/utils.py`
- 11:23 AM - Core model implementation updated:
  - `scratchml/linear_model.py`
  - `scratchml/__pycache__/` updated (import/run activity)
- 11:25 AM - First end-to-end runnable script created/updated:
  - `scripts/run_logreg_synthetic.py`
- Between 11:25 AM and 12:39 PM - Debugged prediction output shape issue (column slice vs row slice) and confirmed correct validation accuracy.
- 12:39 PM - Added convergence comparison experiment:
  - `scripts/compare_gd_newton.py`
- 12:41 PM - Added verification baseline experiment:
  - `scripts/compare_with_torch_baseline.py` (PyTorch `nn.Linear` + `BCEWithLogitsLoss` + LBFGS)
- 12:44 PM - `public/` folder created/updated.
- 12:46 PM - Two extra folders appeared in repo root:
  - `-ErrorAction/`
  - `SilentlyContinue/`
  (These were created during a PowerShell folder-creation command attempt.)
- 12:54 PM - `scripts/` folder updated (new scripts integrated).
- 12:56 PM - Reproducible artifact pipeline created/updated:
  - `scripts/benchmark_all.py`
  - `figures/` folder updated (saved plot image)
  - `results/` folder updated (saved `benchmark.json`, `summary.md`)
  - `scripts/__pycache__/` updated
- 12:57 PM - `report.md` updated.
- 1:00 PM - `README.md` updated.

## What shipped (as of 12/28/2025)
- `scratchml/utils.py` - seeding, splitting, scaling utilities (NumPy-first).
- `scratchml/linear_model.py` - from-scratch logistic regression (GD + Newton).
- `scripts/run_logreg_synthetic.py` - initial synthetic run + loss plot.
- `scripts/compare_gd_newton.py` - GD vs Newton convergence comparison.
- `scripts/compare_with_torch_baseline.py` - verification vs PyTorch baseline (LBFGS).
- `scripts/benchmark_all.py` - one-command end-to-end benchmark producing artifacts:
  - `figures/gd_vs_newton.png`
  - `results/benchmark.json`
  - `results/summary.md`

