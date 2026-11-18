"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints:
  GET  /health          service status
  GET  /models          available models and the default model
  GET  /trips           trip ids the evaluator can choose from
  POST /predict         {"trip_id": "...", "model": optional} -> route + metrics

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.pipeline import (CONFIG, available_trips, gps_points, load_models, predict_trip,
                          route_geometry, score_prediction)
from api.summary import chart_data, dataset_summary, model_rows
from hmm_viterbi.evaluate_hmm import load_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

SPLIT = os.environ.get("PATHSNAP_SPLIT", CONFIG["trip_split"])
MODELS = load_models(CONFIG.get("visible_models"))
HMM_PARAMS = load_params()
DEFAULT_MODEL = CONFIG["default_model"] if CONFIG["default_model"] in MODELS else next(iter(MODELS))
log.info("Loaded models: %s (default: %s) | trip split: %s", list(MODELS), DEFAULT_MODEL, SPLIT)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str
    model: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "split": SPLIT, "models": list(MODELS), "default_model": DEFAULT_MODEL}


@app.get("/models")
def models():
    return {"models": list(MODELS), "default_model": DEFAULT_MODEL}


@app.get("/summary")
def summary():
    """Experiment results for the models on show (aggregate metrics only)."""
    names = list(MODELS)
    return {"default_model": DEFAULT_MODEL, "models": model_rows(names), "dataset": dataset_summary(),
            "charts": chart_data(names, DEFAULT_MODEL)}


@app.get("/trips")
def trips():
    return {"split": SPLIT, "trips": available_trips(SPLIT)}


@app.post("/predict")
def predict(req: PredictRequest):
    name = req.model or DEFAULT_MODEL
    if name not in MODELS:
        raise HTTPException(400, f"unknown model '{name}'")
    if req.trip_id not in available_trips(SPLIT):
        raise HTTPException(404, f"unknown trip '{req.trip_id}'")

    log.info("REQUEST  trip=%s model=%s", req.trip_id, name)
    trip, points, arcs, clf_arcs, seconds = predict_trip(SPLIT, req.trip_id, MODELS[name], HMM_PARAMS)
    metrics = score_prediction(SPLIT, req.trip_id, trip, points, arcs, clf_arcs)
    lines, n_segments = route_geometry(trip, arcs)
    log.info("RESPONSE trip=%s model=%s gps_points=%d segments=%d time=%.2fs top1=%s continuity=%.3f",
             req.trip_id, name, len(trip["track"]), n_segments, seconds,
             None if metrics.get("top1_accuracy") is None else round(metrics["top1_accuracy"], 4),
             metrics.get("route_continuity", float("nan")))
    return {
        "trip_id": req.trip_id,
        "model": name,
        "gps_point_count": len(trip["track"]),
        "gps_points": gps_points(trip, CONFIG["max_gps_points_returned"]),
        "predicted_route": lines,
        "predicted_segments": n_segments,
        "inference_time_s": round(seconds, 3),
        "metrics": metrics,
    }
