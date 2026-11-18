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

```
pathsnap/
├── data/            # raw dataset (train: 80 trips, student_test: 10 trips)
│   └── <trip_id>/   # .track  .nodes  .arcs  .route
├── docs/            # hackathon guideline PDFs
├── src/             # data processing, features, models, HMM/Viterbi
├── notebooks/       # EDA and experiments
├── models/          # saved model weights and configs
├── results/         # experiment logs, CSVs, plots, Folium maps
├── api/             # FastAPI backend (Laptop 1)
├── ui/              # evaluator interface (Laptop 2)
└── report/          # final technical report (PDF)
```

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

## Running

**Laptop 1: API**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Laptop 2: UI**
Open the UI and set the API address to `http://<laptop-1-ip>:8000`.

Both laptops must be on the same network.

## Evaluation rules we follow

- Models are trained only on the 80 training trips. The 10 student-test trips are used only for comparing models.
- Ground-truth routes stay on the server and are never sent to the UI or the model.
- There are no hardcoded predictions. Every result comes from the pipeline.

## Team

- _Add team member names here_

## AI assistance

AI tools were used to help with this project, as permitted by the event's policy. All experiments were run by the team, and all results come from the code in this repository.
