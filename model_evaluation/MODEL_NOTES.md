# PathSnap: Member 4 (Models + Report) — what to push

You push **2 things**:

| Part | Save it in the repo as |
|---|---|
| Part A: Model notes (below) | `model_evaluation/MODEL_NOTES.md` |
| The final report | `report/PathSnap_Report.pdf` |

## Before you start
1. Accept the GitHub collaborator invite from Gagan.
2. Get the code:
   ```
   git clone https://github.com/Gagan-1718/PathSnap-GPS-Map-Matching-Route-Reconstruction.git
   cd PathSnap-GPS-Map-Matching-Route-Reconstruction
   pip install -r requirements.txt
   ```
3. Put the dataset in `data\` as described in `data/README.md`.

## Steps
1. Make your branch: `git switch -c model-notes-and-report`
2. **Check Part A:** confirm every number against the CSVs it lists. Then rewrite it in your own words. Every member must be able to explain these answers in the viva.
3. Save Part A as `model_evaluation/MODEL_NOTES.md`.
4. Use Part A for report section 3 (model comparison). The plots are in `results/plots/` and the maps in `results/maps/`.
5. Push (you can push the report later as a second push):
   ```
   git add model_evaluation/MODEL_NOTES.md
   git commit -m "docs(models): per-model answers with results from the csvs"
   git add report/PathSnap_Report.pdf
   git commit -m "docs: final technical report"
   git push -u origin model-notes-and-report
   ```
6. On GitHub, open a **Pull request** to `main` and tell Gagan.

---

## Part A: `model_evaluation/MODEL_NOTES.md`

# Model notes: answers to the compulsory questions for every model

> **Draft.** Check every number against the files below before you push, and rewrite in your own words. Reports are checked for AI-written text, and every member must be able to explain these answers.
> Sources: `results/training/cv_summary.csv` and `cv_folds.csv`, `results/evaluation/student_test_metrics.csv`, `results/hmm/classifier_vs_hmm.csv`, `results/hpo/*`, `results/model_selection/*`, `saved_models/*_config.json`.

## The task, as the models see it

For every GPS point we list up to 10 nearby road arcs (candidates). Each **(point, candidate)** pair is one row with 10 features: distance, distance rank, gap to the nearest candidate, heading difference, heading-reliable flag, speed, arc length, position along the arc, number of candidates, and time gap. The model answers **"is this the true road? (1/0)"** with a probability. For each point, the candidate with the highest probability is the model's pick (**top-1**). The HMM then uses these probabilities to build a connected route.

- Training: 80 trips, 5-fold **GroupKFold by trip** (a trip is never in both train and validation).
- Testing: 10 student-test trips.
- "Train" scores below were measured on 10 random training trips per fold, to compare against validation for overfitting.

## Summary table

| Model | Family | CV Macro F1 (± std) | CV Top-1 | CV Log loss | Train Macro F1 | Test Top-1 (model only) | Test Top-1 + HMM | Test Macro F1 | Test Log loss |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.908 ± 0.020 | 87.8% | 0.136 | 0.916 | 87.9% | 88.9% | 0.910 | 0.124 |
| Gaussian Naive Bayes | Probabilistic | 0.860 ± 0.013 | 86.9% | 0.344 | 0.870 | 86.3% | 88.5% | 0.854 | 0.383 |
| Decision Tree | Tree | 0.929 ± 0.017 | 89.1% | 0.093 | 0.941 | 90.1% | 91.7% | 0.935 | 0.081 |
| KNN | Instance-based | 0.921 ± 0.018 | 88.0% | 0.157 | 1.000 | 89.5% | 90.7% | 0.930 | 0.127 |
| Random Forest | Ensemble (bagging) | 0.930 ± 0.017 | 89.1% | 0.091 | 0.946 | 90.2% | 91.5% | 0.937 | 0.079 |
| LightGBM | Boosting | 0.929 ± 0.016 | 89.1% | 0.090 | 0.952 | 90.3% | 91.4% | 0.937 | 0.077 |
| **LightGBM, tuned (final)** | Boosting | **0.931 ± 0.016** | **89.2%** | **0.089** | — | 90.2% | 91.6% | 0.937 | 0.078 |

"+ HMM" uses the locked decoder settings (`saved_models/hmm_params.json`).

---

## 1. Logistic Regression
**A1. Why suitable?** It's a simple baseline that outputs a probability directly, which the HMM needs.
**A2. Strengths / limits.** Fast (2.3 s to train per fold) with reasonable probabilities. But the boundary is a straight line, so it can't learn combinations like "close **and** pointing the same way" unless we build them as features. That's why it trails the tree models by about 2 points.
**B5. Hyperparameters.** `C = 1.0` (inverse regularisation strength), `max_iter = 1000`. Features are standardised (the scaler is fitted on training folds only).
**B6. Loss.** Binary cross-entropy (log loss) with L2 regularisation.
**C7–8. Test results.** Top-1 87.9% (88.9% with HMM), Macro F1 0.910, log loss 0.124.
**C9. Over/underfitting?** Train 0.916 vs validation 0.908, a very small gap. Both are lower than the tree models, so this is **mild underfitting**: the model is too simple, not memorising.

## 2. Gaussian Naive Bayes
**A1.** A fast probabilistic baseline from a different family.
**A2.** It trains almost instantly (0.3 s). But it assumes every feature is independent and bell-shaped within each class. Here features are strongly related (for example distance, distance rank and distance gap), so that assumption is broken.
**B5.** `var_smoothing = 1e-9` (default).
**B6.** No iterative loss. It estimates each class's mean and variance per feature by maximum likelihood.
**C7–8.** Top-1 86.3% (88.5% with HMM), Macro F1 0.854. It has the worst log loss (0.383), meaning badly calibrated probabilities. Its recall is high (0.906) but its precision is low (0.652): it marks too many candidates as correct.
**C9.** Train 0.870 vs validation 0.860. **Underfitting**: both scores are the lowest of all models.

## 3. Decision Tree
**A1.** It learns yes/no rules such as "distance < x m and heading difference < y°", which match how a person would pick the road.
**A2.** It captures non-linear rules, needs no scaling, and is very fast (0.0015 ms per row at test time). But it overfits when too deep, and its probabilities are step-like (one value per leaf).
**B5.** `max_depth = 8`, `min_samples_leaf = 50`. Bayesian tuning (20 trials) found `max_depth = 8, min_samples_leaf = 14, criterion = entropy`, which did **not** beat these settings (CV Macro F1 0.9284 vs 0.9288).
**B6.** Splits chosen by **Gini impurity**.
**C7–8.** Top-1 90.1% (91.7% with HMM), Macro F1 0.935, log loss 0.081.
**C9.** Train 0.941 vs validation 0.929, a small gap. **Slight overfitting, kept under control** by the depth and leaf-size limits.

## 4. K-Nearest Neighbours (KNN)
**A1.** A different idea: judge a candidate by the 25 most similar training candidates.
**A2.** There's almost no training step. But prediction is very slow, because each row is compared with over a million stored training rows: 133 s per CV fold and 3.6 ms per row at test time, roughly 370× slower than Random Forest and 2,000× slower than the Decision Tree. The saved model is 210 MB, too big for GitHub.
**B5.** `n_neighbors = 25`, `weights = distance`, features standardised.
**B6.** None. It's a lazy learner (distance-weighted vote).
**C7–8.** Top-1 89.5% (90.7% with HMM), Macro F1 0.930, log loss 0.127.
**C9.** Train 1.000 vs validation 0.921. **Overfitting / memorisation**: with distance weighting, every training row is its own nearest neighbour, so the training score is always perfect.

## 5. Random Forest
**A1.** It averages many trees, which fixes the single tree's instability.
**A2.** Stable and strong (highest CV Macro F1 among untuned models). But it's slower to train (27.5 s per fold), has a larger file (about 20 MB) and is harder to explain.
**B5.** `n_estimators = 100`, `max_depth = 12`, `min_samples_leaf = 20`.
**B6.** Gini impurity in each tree, with bootstrap sampling (bagging).
**C7–8.** Top-1 90.2% (91.5% with HMM), Macro F1 0.937, log loss 0.079.
**C9.** Train 0.946 vs validation 0.930. A **small gap**: averaging keeps overfitting low.

## 6. LightGBM (final model after tuning)
**A1.** Gradient boosting builds trees one after another, each fixing the previous errors. It's known to work well on table-style features like ours.
**A2.** Best-calibrated probabilities (lowest log loss) and fast to train (4.3 s per fold). But it has more settings to tune and is harder to explain.
**B5 (untuned).** `n_estimators = 200`, `learning_rate = 0.1`, `num_leaves = 31`, `min_child_samples = 50`.
**B5 (tuned, Optuna TPE, 20 trials).** `n_estimators = 218`, `learning_rate ≈ 0.024`, `num_leaves = 34`, `min_child_samples = 79`, `colsample_bytree ≈ 0.76`, `reg_lambda ≈ 0.21`.
**B6.** Binary log loss, minimised by gradient boosting, with L2 regularisation (`reg_lambda`).
**C7–8.** Tuned: test Top-1 90.2% (91.6% with HMM), Macro F1 0.937, log loss 0.078.
**C9.** Untuned: train 0.952 vs validation 0.929, a moderate gap. Tuning chose a smaller learning rate, more samples per leaf and feature subsampling, which all reduce overfitting. CV log loss improved from 0.0895 to 0.0886.

---

## Why tuned LightGBM is the final model

1. **Full pipeline (model + HMM) on all 80 training trips** (`results/model_selection/`, run with the earlier HMM settings, before the final decoder re-tune): Random Forest 91.73%, tuned LightGBM 91.65%, LightGBM 91.63%, Decision Tree 91.63%.
2. **Paired bootstrap** (2,000 resamples of trips): Random Forest beats tuned LightGBM in only 76% of resamples, below our 90% bar. So it's a **statistical tie**.
3. **Tie-break, fixed in advance:** better probability quality, because the HMM uses the probabilities. Tuned LightGBM has the lowest CV log loss (0.0886).
4. The 10 student-test trips only **confirm** the choice; they weren't used to make it.

## HMM effect (test trips, all models)
- The HMM raises right-road accuracy by about 1–2 points for every model.
- Route continuity goes from 68–74% (model only) to 89–92% (with HMM).
