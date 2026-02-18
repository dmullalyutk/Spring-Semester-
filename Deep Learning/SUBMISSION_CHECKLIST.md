# Submission Checklist - Group Assignment 1

## 1. Final Code State
- [x] `incremental_nn.py` uses incremental mini-batch learning
- [x] 3 hidden layers with sigmoid hidden activations
- [x] `pricing_test.csv` not used for training/tuning logic
- [x] Primary config locked (`log_shift_diff`, `sku_scale=0.0`, `category_scale=0.55`, `x_clip=7.0`)

## 2. Required Deliverables
- [x] Learning curve plot: `learning_curve.png`
- [x] Variable importance plot: `variable_importance.png`
- [x] Partial dependence plots: `partial_dependence.png`
- [x] RAM usage reported in logs/output
- [x] Training time reported in logs/output
- [x] Experiment/run log: `EXPERIMENT_CHANGELOG.md`
- [x] Contribution statement: `CONTRIBUTION_STATEMENT.txt`
- [x] Runbook: `RUNBOOK.txt`

## 3. Final Reporting Run (Before Packaging)
- [x] Set `run_interpretability = True` in `incremental_nn.py`
- [x] Run: `py -3.14 "Deep Learning/incremental_nn.py"`
- [x] Confirm final metrics and regenerated plots (`run_final_artifacts_seed42_20260218_120858.log`)
- [x] Set `run_interpretability = False` after final run

## 4. Presentation (PDF/PPT)
- [ ] Problem + objective
- [ ] Data + features
- [ ] Model architecture + incremental training method
- [ ] Final config and why selected
- [ ] Learning curve, variable importance, PDPs
- [ ] Final R2/RMSE + training time + max RAM
- [ ] Compliance statement: no test tuning
- [ ] Team contributions summary

## 5. Archive Packaging
- [ ] Folder name includes your name + group number
- [ ] Include code + presentation + runbook + contribution statement
- [ ] Include required output images
- [ ] Zip archive created and validated

## 6. Optional Robustness Note
- [x] 3-seed robustness runs completed (seed 41/52/63)
- [ ] Add mean/std seed result summary to presentation

