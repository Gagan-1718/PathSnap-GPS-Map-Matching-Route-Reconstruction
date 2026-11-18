"""Interactive Folium maps: GPS points, road network, reference and predicted route.

Output: results/maps/<split>_<trip_id>.html
Run:    python -m visualisations.folium_maps [model_name]
"""
import sys

import folium
import joblib
import numpy as np

from data_processing.load_data import ROOT, prepare_trip
from feature_engineering.candidates import generate_candidates
from feature_engineering.features import FEATURES, add_features
from hmm_viterbi.evaluate_hmm import classifier_only, load_params
from hmm_viterbi.hmm import decode_trip

OUT = ROOT / "results" / "maps"

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
     padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font: 13px sans-serif;">
<b>Trip {trip}</b><br>
<span style="color:#9e9e9e">&#9644;</span> Road network (near track)<br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS points<br>
<span style="color:#2ca02c">&#9644;</span> Reference route<br>
<span style="color:#d62728">&#9644;</span> Predicted route ({method})<br>
<span style="color:#ff7f0e">&#9644;</span> Classifier-only prediction<br>
</div>"""


def arc_lines(trip, arc_ids):
    nodes = trip["nodes"][["lat", "lon"]].to_numpy()
    a = trip["arcs"].to_numpy()
    return [[nodes[a[i, 0]].tolist(), nodes[a[i, 1]].tolist()] for i in arc_ids]


def route_polylines(trip, arc_seq):
    """Join consecutive connected arcs into polylines (breaks where not connected)."""
    nodes = trip["nodes"][["lat", "lon"]].to_numpy()
    a = trip["arcs"].to_numpy()
    seq = np.asarray(arc_seq)
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    lines, cur = [], []
    for i, arc in enumerate(seq):
        if cur and a[seq[i - 1], 1] != a[arc, 0]:
            lines.append(cur)
            cur = []
        if not cur:
            cur.append(nodes[a[arc, 0]].tolist())
        cur.append(nodes[a[arc, 1]].tolist())
    if cur:
        lines.append(cur)
    return lines


def trip_map(split, trip_id, model_name=None):
    trip = prepare_trip(split, trip_id)
    tr = trip["track"]
    cands = add_features(generate_candidates(trip), trip)
    m = folium.Map(tiles="cartodbpositron", control_scale=True)
    m.fit_bounds([[tr["lat"].min(), tr["lon"].min()], [tr["lat"].max(), tr["lon"].max()]])

    net = folium.FeatureGroup("Road network (near track)")
    for line in arc_lines(trip, np.unique(cands["arc"])):
        folium.PolyLine(line, color="#9e9e9e", weight=2, opacity=0.6).add_to(net)
    net.add_to(m)

    gps = folium.FeatureGroup("Raw GPS points")
    step = max(1, len(tr) // 1500)
    for lat, lon, t in tr[["lat", "lon", "t"]].to_numpy()[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True, opacity=0.8,
                            tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)

    if trip.get("route") is not None:
        ref = folium.FeatureGroup("Reference route")
        for line in route_polylines(trip, trip["route"]):
            folium.PolyLine(line, color="#2ca02c", weight=6, opacity=0.7).add_to(ref)
        ref.add_to(m)

    method = "none"
    if model_name:
        model = joblib.load(ROOT / "saved_models" / f"{model_name}.joblib")
        prob = model.predict_proba(cands[FEATURES].to_numpy())[:, 1]
        _, clf = classifier_only(cands, prob)
        layer = folium.FeatureGroup("Classifier-only prediction", show=False)
        for line in route_polylines(trip, clf):
            folium.PolyLine(line, color="#ff7f0e", weight=3, opacity=0.9).add_to(layer)
        layer.add_to(m)
        _, pred = decode_trip(cands, prob, trip["arcs"], tr["t"].to_numpy(), load_params())
        layer = folium.FeatureGroup("Predicted route (HMM)")
        for line in route_polylines(trip, pred):
            folium.PolyLine(line, color="#d62728", weight=3, opacity=0.9).add_to(layer)
        layer.add_to(m)
        method = f"{model_name} + HMM"

    folium.Marker(tr[["lat", "lon"]].iloc[0].tolist(), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(tr[["lat", "lon"]].iloc[-1].tolist(), tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(trip=trip_id, method=method)))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main(model_name=None):
    OUT.mkdir(parents=True, exist_ok=True)
    # EDA map: a training trip with GPS, network and reference route
    trip_map("train", "00000005").save(OUT / "train_00000005.html")
    # Prediction maps on student-testing trips
    for tid in ["00000007", "00000052", "00000065"]:
        trip_map("student_test", tid, model_name).save(OUT / f"student_test_{tid}.html")
    print("maps saved to", OUT)


if __name__ == "__main__":
    main(*sys.argv[1:])
