# PathSnap â€“ GPS Map-Matching & Route Reconstruction

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

**Google Colab:** open [PathSnap_Colab.ipynb](PathSnap_Colab.ipynb) â†’
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gagan-1718/PathSnap-GPS-Map-Matching-Route-Reconstruction/blob/feat/pathsnap-pipeline/PathSnap_Colab.ipynb).
It clones this repo, asks you to upload the dataset zip, and runs every stage below with its outputs.

**Locally:** run from the repository root, in this order. You can also run `deployment\run_pipeline.bat` to do all of it.

```bash
python -m data_processing.explore_data            # dataset inventory and quality checks
python -m feature_engineering.build_dataset       # candidates, features, labels
python -m model_training.train_models             # 5 models, 5-fold GroupKFold CV
python -m model_evaluation.evaluate_models        # student-test comparison
python -m hmm_viterbi.tune_decoder random_forest  # decoder penalties (training trips only)
python -m hyperparameter_optimisation.bayes_opt random_forest 20
python -m hmm_viterbi.evaluate_hmm                # classifier-only vs classifier + HMM
python -m visualisations.plots                    # plots in results/plots
python -m visualisations.folium_maps random_forest_tuned   # maps in results/maps
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

| # | Model | Family |
|---|---|---|
| 1 | Logistic Regression | Linear / probabilistic |
| 2 | Decision Tree | Tree-based |
| 3 | Gaussian Naive Bayes | Probabilistic |
| 4 | K-Nearest Neighbours | Instance-based |
| 5 | Random Forest | Ensemble |

Results are in `results/training/cv_summary.csv`, `results/evaluation/student_test_metrics.csv` and `results/hmm/classifier_vs_hmm.csv`.

## Evaluation rules we follow

- Models are trained only on the 80 training trips. The 10 student-test trips are used only for comparing models.
- Ground-truth routes stay on the server and are never sent to the UI or the model.
- There are no hardcoded predictions. Every result comes from the pipeline.

## Team

- _Add team member names here_

## AI assistance

AI tools were used to help with this project, as permitted by the event's policy. All experiments were run by the team, and all results come from the code in this repository.
