# PathSnap-GPS-Map-Matching-Route-Reconstruction
PathSnap snaps noisy GPS observations to nearby road segments and reconstructs a
connected route using candidate probabilities and Viterbi decoding.

Built for the UE24CS352A Machine Learning Hackathon (PES University, RR Campus).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

Place each trip in `data/<trip_id>/` with a `.track`, `.nodes`, `.arcs`, and
optional `.route` file. Track rows are `longitude, latitude, time`; node rows
are `longitude, latitude`; arc rows are `from_node, to_node`; route rows contain
arc IDs.

## Run

Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `ui/index.html` and set the API address to
`http://<laptop-1-ip>:8000`.
