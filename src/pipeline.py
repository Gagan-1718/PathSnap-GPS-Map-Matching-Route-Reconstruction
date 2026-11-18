"""End-to-end GPS route reconstruction pipeline."""

from pathlib import Path

import numpy as np

from .features import extract_features, find_candidates
from .model import CandidateClassifier
from .viterbi import ViterbiDecoder


class PathSnapPipeline:
    def __init__(self, classifier=None, radius_m=100.0, max_candidates=10):
        self.classifier = classifier or CandidateClassifier()
        self.radius_m = radius_m
        self.max_candidates = max_candidates
        self.decoder = ViterbiDecoder()

    def predict(self, track, nodes, arcs):
        candidates = find_candidates(
            track, nodes, arcs, radius_m=self.radius_m, max_candidates=self.max_candidates
        )
        feature_rows = extract_features(track, nodes, arcs, candidates)
        emissions = []
        for row in feature_rows:
            if row:
                emissions.append(self.classifier.positive_probabilities(row))
            else:
                emissions.append(np.asarray([]))
        route = self.decoder.decode(
            [[1.0] * len(row) if len(row) else [] for row in feature_rows],
            candidates,
            arcs,
        )
        if emissions and candidates:
            route = self.decoder.decode(emissions, candidates, arcs)
        return route, candidates
