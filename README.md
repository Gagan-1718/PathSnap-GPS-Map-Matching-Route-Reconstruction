# PathSnap – GPS Map-Matching & Route Reconstruction

Noisy GPS points often drift off the road. PathSnap snaps each GPS point to the road segment the vehicle was really on.
It uses a machine-learning classifier to score nearby road segments, then an HMM + Viterbi decoder to turn those scores into one connected route.

Built for the UE24CS352A Machine Learning Hackathon (PES University, RR Campus).

## How it works

1. **Read the data**: GPS track, road nodes, road segments and reference route for each trip.
2. **Find candidates**: for every GPS point, list the road segments nearby.
3. **Score candidates**: a trained classifier gives each candidate a probability of being the right road.
4. **Decode the route**: HMM + Viterbi picks the most likely sequence of connected segments.
5. **Serve and show**: a FastAPI backend returns the route and its accuracy, and a web UI draws it on a map.

## Folder structure

Folders follow the required deliverable components.

| Folder | Deliverable | Main files |
|---|---|---|
| `data_processing/` | Read, inspect, clean, preprocess | `load_data.py`, `explore_data.py` |
| `feature_engineering/` | Candidate generation, features, labels | `candidates.py`, `features.py`, `build_dataset.py` |
| `model_training/` | Training for every model | `train_models.py` |
| `model_evaluation/` | Metrics, model comparison | `metrics.py`, `evaluate_models.py` |
| `hyperparameter_optimisation/` | Bayesian optimisation (Optuna TPE) | `bayes_opt.py` |
| `hmm_viterbi/` | Route reconstruction and decoding | `hmm.py`, `tune_decoder.py`, `evaluate_hmm.py` |
| `api/` | FastAPI backend and model loading | `main.py`, `pipeline.py` |
| `ui/` | Evaluator interface | `index.html` |
| `deployment/` | Config, run scripts, instructions | `config.json`, `run_api.bat`, `run_ui.bat`, `DEPLOYMENT.md` |
| `visualisations/` | Folium maps and plots | `folium_maps.py`, `plots.py` |
| `saved_models/` | Trained weights and configs | `*.joblib`, `*_config.json`, `hmm_params.json` |
| `results/` | CSVs, logs, plots, maps | `eda/`, `features/`, `training/`, `evaluation/`, `hpo/`, `hmm/`, `plots/`, `maps/` |
| `data/` | Raw dataset (not committed) | see `data/README.md` |

## Data format

| File     | Columns                           | Meaning                                  |
|----------|-----------------------------------|------------------------------------------|
| `.track` | longitude, latitude, time (s)     | GPS points, about 1 per second           |
| `.nodes` | longitude, latitude               | Road points (row number = node id)       |
| `.arcs`  | from_node, to_node                | Road segments (row number = arc id)      |
| `.route` | arc_id                            | Reference route, in driving order        |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then place the dataset in `data/` (see [data/README.md](data/README.md)).

## Reproducing the results

**Google Colab:** open [PathSnap_Colab.ipynb](PathSnap_Colab.ipynb) →
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gagan-1718/PathSnap-GPS-Map-Matching-Route-Reconstruction/blob/feat/pathsnap-pipeline/PathSnap_Colab.ipynb).
It clones this repo, asks you to upload the dataset zip, and runs every stage below with its outputs.

**Locally:** run from the repository root, in this order. You can also run `deployment\run_pipeline.bat` to do all of it.

```bash
python -m data_processing.explore_data            # dataset inventory and quality checks
python -m feature_engineering.build_dataset       # candidates, features, labels
python -m model_training.train_models             # 6 models, 5-fold GroupKFold CV
python -m model_evaluation.evaluate_models        # student-test comparison
python -m hmm_viterbi.tune_decoder random_forest  # decoder penalties (training trips only)
python -m hyperparameter_optimisation.bayes_opt lightgbm 20
python -m model_evaluation.pipeline_cv decision_tree random_forest lightgbm lightgbm_tuned   # model selection
python -m hmm_viterbi.evaluate_hmm                # classifier-only vs classifier + HMM
python -m visualisations.plots                    # plots in results/plots
python -m visualisations.folium_maps lightgbm_tuned   # maps in results/maps
```

## Running the system (two laptops)

**Laptop 1: API**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Laptop 2: UI**
```bash
cd ui && python -m http.server 5500     # then open http://localhost:5500
```
Enter `http://<laptop-1-ip>:8000`, press **Connect** and pick a trip.

Full steps, including the firewall and evaluator trips, are in [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md).

## Models compared

"Right road" means the share of GPS points matched to the correct road on the 10 student-test trips. CV uses 80 training trips with 5-fold GroupKFold by trip.

| # | Model | Family | CV right road | Test: model only | Test: + HMM | Connected route + HMM |
|---|---|---|---|---|---|---|
| 1 | Logistic Regression | Linear | 87.8% | 87.9% | 88.9% | 91.8% |
| 2 | Gaussian Naive Bayes | Probabilistic | 86.9% | 86.3% | 88.5% | 89.6% |
| 3 | Decision Tree | Tree | 89.1% | 90.1% | 91.7% | 89.5% |
| 4 | K-Nearest Neighbours | Instance-based | 88.0% | 89.5% | 90.7% | 89.3% |
| 5 | Random Forest | Ensemble (bagging) | 89.1% | 90.2% | 91.5% | 90.0% |
| 6 | LightGBM | Boosting | 89.1% | 90.3% | 91.4% | 89.3% |
| ★ | **LightGBM, tuned (final)** | Boosting | **89.2%** | 90.2% | 91.6% | 89.6% |

"+ HMM" columns use the final, locked decoder settings (`saved_models/hmm_params.json`).

### How the final model was chosen

The rule was fixed before looking at the results:

1. **Main score:** right road **with HMM** over the full pipeline, cross-validated on all **80 training trips** (`results/model_selection/`; run with the earlier decoder settings, before the final decoder re-tune).
2. **Is the gap real?** A paired bootstrap over trips (2000 resamples). A model must win at least 90% of resamples to count as better.
3. **If tied:** prefer better probability quality (log loss), because the HMM uses the probabilities. Then prefer speed.

| Full pipeline, 80 trips | Right road + HMM | Log loss (CV) |
|---|---|---|
| Random Forest | 91.73% | 0.0905 |
| **LightGBM, tuned** | 91.65% | **0.0886** |
| LightGBM | 91.63% | 0.0895 |
| Decision Tree | 91.63% | 0.0926 |

The top models are a **statistical tie**: Random Forest beats tuned LightGBM in only 76% of bootstrap resamples. The tie-break picks **tuned LightGBM**, which has the best probabilities, the best route match (0.904) and the best model-only accuracy (89.2%). Bayesian optimisation (Optuna TPE, 20 trials) improved LightGBM's CV score and log loss. See `results/hpo/`. The 10 student-test trips only confirm the choice; they were not used to make it.

**HMM + Viterbi** improves every model: route continuity rises from about 68–74% (model only) to 89–92%.

### HMM decoder settings (locked)

The decoder was re-tuned with the final model's probabilities on 15 training trips, using a small grid to avoid overfitting. It was then checked on the **65 other training trips** and the 10 test trips (`results/hmm/decoder_comparison.csv`):

| Unseen trips | Right road: old → new | Connected route: old → new |
|---|---|---|
| 65 training trips | 91.89% → 92.00% | 85.9% → 88.5% |
| 10 student-test trips | 91.62% → 91.55% | 87.0% → 89.6% |

Accuracy is unchanged (within noise), and continuity improves by 2.6 points on both unseen sets, so the new settings were kept: emission weight 0.5, hop penalty 1, U-turn penalty 1, disconnect penalty 35, gap reset after 10 s.

Sources: `results/training/cv_summary.csv`, `results/evaluation/student_test_metrics.csv`, `results/hmm/classifier_vs_hmm.csv`.
KNN's model file (210 MB) is too large for GitHub. Rebuild it with `python -m model_training.train_models knn`.

## Evaluation rules we follow

- Models are trained only on the 80 training trips. The 10 student-test trips are used only for comparing models.
- Ground-truth routes stay on the server and are never sent to the UI or the model.
- There are no hardcoded predictions. Every result comes from the pipeline.

## Team

- _Add team member names here_

## AI assistance

AI tools were used to help with this project, as permitted by the event's policy. All experiments were run by the team, and all results come from the code in this repository.
