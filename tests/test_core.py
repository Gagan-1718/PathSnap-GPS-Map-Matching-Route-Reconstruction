import tempfile
from pathlib import Path

import numpy as np

from src.data_loader import load_arcs, load_nodes, load_track, load_trip, load_route
from src.features import find_candidates
from src.model import CandidateClassifier
from src.viterbi import ViterbiDecoder


def test_load_trip_and_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        trip = Path(tmp)
        np.savetxt(trip / "gps.track", [[0.0, 0.0, 0], [0.001, 0.0, 1]])
        np.savetxt(trip / "map.nodes", [[0.0, 0.0], [0.002, 0.0]])
        np.savetxt(trip / "roads.arcs", [[0, 1]])
        np.savetxt(trip / "reference.route", [0])

        data = load_trip(trip)
        candidates = find_candidates(data["track"], data["nodes"], data["arcs"], radius_m=300)

        assert data["track"].shape == (2, 3)
        assert data["route"].tolist() == [0]
        assert len(candidates) == 2
        assert all(candidate[0].arc_id == 0 for candidate in candidates)


def test_viterbi_returns_connected_arc_sequence():
    nodes = np.array([[0, 0], [1, 0], [2, 0]])
    arcs = np.array([[0, 1], [1, 2]])
    candidate_rows = [
        [type("C", (), {"arc_id": 0})()],
        [type("C", (), {"arc_id": 1})()],
    ]
    decoder = ViterbiDecoder()
    route = decoder.decode([[0.9], [0.9]], candidate_rows, arcs)
    assert route == [0, 1]


def test_untrained_classifier_has_uniform_fallback():
    classifier = CandidateClassifier()
    probabilities = classifier.positive_probabilities([[1.0], [2.0]])
    np.testing.assert_allclose(probabilities, [0.5, 0.5])
