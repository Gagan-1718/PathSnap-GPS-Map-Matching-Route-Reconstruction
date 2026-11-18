"""HMM + Viterbi route reconstruction.

Hidden states : the candidate arcs of each GPS point.
Emission      : log of the classifier probability that the candidate is correct.
Transition    : based on road-network connectivity between consecutive arcs
                (same arc, directly connected, or connected through 1-2 arcs);
                arcs that are not connected get a large penalty instead of 0
                probability, so the decoder never gets stuck.
Large GPS gaps: if the time gap is above GAP_S, transitions are made uniform,
                because the vehicle may have travelled far in between.
Viterbi       : dynamic programming that keeps, for every candidate, the best
                score of any path ending there, then backtracks the best path.
"""
from collections import defaultdict

import numpy as np

DEFAULT_PARAMS = {
    "emission_weight": 1.0,
    "hop_penalty": 1.0,         # cost per extra arc travelled between two points
    "uturn_penalty": 3.0,       # cost of turning back onto the reverse arc
    "disconnect_penalty": 12.0, # cost of jumping to an unconnected arc
    "gap_s": 10.0,              # time gap after which transitions become uniform
    "max_hops": 3,
}


class RoadGraph:
    """Arc connectivity helper: number of arc hops between two arcs."""

    def __init__(self, arcs):
        a = arcs.to_numpy() if hasattr(arcs, "to_numpy") else np.asarray(arcs)
        self.u, self.v = a[:, 0], a[:, 1]
        self.out_nodes = defaultdict(set)   # node -> nodes reachable by one arc
        for s, e in zip(self.u.tolist(), self.v.tolist()):
            self.out_nodes[s].add(e)
        self._reach2 = {}
        self._cache = {}

    def _two_step(self, node):
        if node not in self._reach2:
            r = set()
            for n in self.out_nodes.get(node, ()):
                r |= self.out_nodes.get(n, set())
            self._reach2[node] = r
        return self._reach2[node]

    def hops(self, a, b):
        """0 same arc, 1 b starts where a ends, 2/3 via one/two arcs, inf otherwise."""
        key = (a, b)
        if key in self._cache:
            return self._cache[key]
        if a == b:
            h = 0
        else:
            end, start = self.v[a], self.u[b]
            if end == start:
                h = 1
            elif start in self.out_nodes.get(end, ()):
                h = 2
            elif start in self._two_step(end):
                h = 3
            else:
                h = np.inf
        self._cache[key] = h
        return h

    def is_uturn(self, a, b):
        return self.u[a] == self.v[b] and self.v[a] == self.u[b]


def transition_matrix(graph, prev_arcs, cur_arcs, dt, params):
    if dt > params["gap_s"]:
        return np.zeros((len(prev_arcs), len(cur_arcs)))
    m = np.empty((len(prev_arcs), len(cur_arcs)))
    for i, a in enumerate(prev_arcs):
        for j, b in enumerate(cur_arcs):
            h = graph.hops(a, b)
            if h > params["max_hops"]:
                m[i, j] = -params["disconnect_penalty"]
            else:
                m[i, j] = -params["hop_penalty"] * max(h - 1, 0) * (h > 0) - (
                    params["uturn_penalty"] if h == 1 and graph.is_uturn(a, b) else 0.0)
    return m


def viterbi(point_ids, cand_arcs, cand_probs, times, graph, params=None):
    """Decode the best arc sequence.

    point_ids  : GPS point index for each step (sorted)
    cand_arcs  : list of arrays, candidate arc ids per step
    cand_probs : list of arrays, classifier probabilities per step
    times      : timestamp per step
    Returns an array with one predicted arc per step.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    emit = [p["emission_weight"] * np.log(np.clip(pr, 1e-6, 1.0)) for pr in cand_probs]
    score = emit[0].copy()
    back = []
    for t in range(1, len(point_ids)):
        trans = transition_matrix(graph, cand_arcs[t - 1], cand_arcs[t], times[t] - times[t - 1], p)
        total = score[:, None] + trans
        best_prev = total.argmax(axis=0)
        score = total[best_prev, np.arange(total.shape[1])] + emit[t]
        back.append(best_prev)
    path = np.empty(len(point_ids), dtype=int)
    k = int(score.argmax())
    path[-1] = cand_arcs[-1][k]
    for t in range(len(point_ids) - 1, 0, -1):
        k = int(back[t - 1][k])
        path[t - 1] = cand_arcs[t - 1][k]
    return path


def decode_trip(cands, prob, arcs, times, params=None, graph=None):
    """Run Viterbi on one trip's candidate table.

    cands : DataFrame with columns point, arc (one trip)
    prob  : classifier probabilities aligned with cands rows
    arcs  : the trip's arcs DataFrame
    times : GPS timestamps indexed by point
    Returns (points, predicted_arcs).
    """
    d = cands[["point", "arc"]].copy()
    d["prob"] = prob
    d = d.sort_values(["point", "prob"], ascending=[True, False])
    groups = d.groupby("point", sort=True)
    points = np.array(list(groups.groups.keys()))
    cand_arcs = [g["arc"].to_numpy() for _, g in groups]
    cand_probs = [g["prob"].to_numpy() for _, g in groups]
    graph = graph or RoadGraph(arcs)
    path = viterbi(points, cand_arcs, cand_probs, np.asarray(times)[points], graph, params)
    return points, path
