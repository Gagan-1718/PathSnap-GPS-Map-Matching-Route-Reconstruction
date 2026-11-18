"""FastAPI service for route reconstruction."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import load_trip
from src.pipeline import PathSnapPipeline


app = FastAPI(title="PathSnap", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = PathSnapPipeline()


class RouteRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    data_dir = Path("pathsnap/data")
    if not data_dir.exists():
        data_dir = Path("data")
    if not data_dir.exists():
        return {"trips": []}
    return {"trips": sorted(p.name for p in data_dir.iterdir() if p.is_dir())}


@app.post("/route")
def reconstruct(request: RouteRequest):
    data_dir = Path("pathsnap/data") if Path("pathsnap/data").exists() else Path("data")
    trip_dir = data_dir / request.trip_id
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail="Trip not found")
    trip = load_trip(trip_dir)
    route, candidates = pipeline.predict(trip["track"], trip["nodes"], trip["arcs"])
    response = {"trip_id": request.trip_id, "route": route}
    if trip["route"] is not None:
        reference = trip["route"].tolist()
        correct = sum(a == b for a, b in zip(route, reference))
        response["accuracy"] = correct / max(len(route), 1)
    return response
