"""HMM/Viterbi decoding for a connected road-arc route."""

import numpy as np


class ViterbiDecoder:
    def __init__(self, connectivity_threshold=1):
        self.connectivity_threshold = connectivity_threshold

    def _can_connect(self, from_arc, to_arc, arcs, max_gap=1):
        from_to = int(arcs[from_arc][1])
        to_from = int(arcs[to_arc][0])
        return from_to == to_from

    def decode(self, emissions, candidates, arcs):
        n_points = len(candidates)
        if n_points == 0:
            return []

        lengths = [len(row) for row in candidates]
        if any(length == 0 for length in lengths):
            return []
        if len(emissions) != n_points:
            raise ValueError("Emissions must have one row per GPS point")

        scores = np.full((n_points, max(lengths)), -np.inf)
        back = np.full((n_points, max(lengths)), -1, dtype=int)
        for j in range(lengths[0]):
            score = emissions[0][j] if hasattr(emissions[0], "__getitem__") else emissions[0]
            scores[0, j] = np.log(max(float(score), 1e-12))

        for i in range(1, n_points):
            for j in range(lengths[i]):
                best_score = -np.inf
                best_previous = -1
                arc_id = candidates[i][j].arc_id
                for k in range(lengths[i - 1]):
                    previous_arc = candidates[i - 1][k].arc_id
                    connected = self._can_connect(previous_arc, arc_id, arcs)
                    transition = 0.0 if connected else -12.0
                    score = scores[i - 1, k] + transition
                    if score > best_score:
                        best_score, best_previous = score, k
                emission = emissions[i][j]
                emission = max(float(emission), 1e-12)
                scores[i, j] = best_score + np.log(emission)
                back[i, j] = best_previous

        route = [-1] * n_points
        route[-1] = int(np.argmax(scores[-1, :lengths[-1]]))
        for i in range(n_points - 1, 0, -1):
            route[i - 1] = back[i, route[i]]
        return [candidates[i][route[i]].arc_id for i in range(n_points)]

    def decode_route(self, emissions, candidates, arcs):
        return self.decode(emissions, candidates, arcs)
