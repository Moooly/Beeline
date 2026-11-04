"""Deterministic, bounded reduction of signed regulatory edges.

Rank by decreasing magnitude, then source and target names. For repeated edges
keep the largest magnitude (positive wins an exact sign tie). Self-links and
nonfinite scores never consume regulator slots.
"""
import heapq
from math import isfinite

import numpy as np
import pandas as pd

COLUMNS = ['Gene1', 'Gene2', 'EdgeWeight']


class _Candidate:
    __slots__ = ('source', 'weight')

    def __init__(self, source, weight):
        self.source, self.weight = source, weight

    def __lt__(self, other):
        # The root is the worst retained regulator.
        if abs(self.weight) != abs(other.weight):
            return abs(self.weight) < abs(other.weight)
        return self.source > other.source


def reduce_edges(streams, top_k=None):
    targets = {}
    for stream in streams:
        for source, target, raw_weight in stream:
            if pd.isna(source) or pd.isna(target):
                continue
            source, target = str(source), str(target)
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            if not source or not target or source == target or not isfinite(weight):
                continue
            heap, live = targets.setdefault(target, ([], {}))
            previous = live.get(source)
            if previous is not None:
                if (abs(weight), weight) <= (abs(previous.weight), previous.weight):
                    continue
            candidate = _Candidate(source, weight)
            if top_k is None:
                live[source] = candidate
                continue
            while heap and live.get(heap[0].source) is not heap[0]:
                heapq.heappop(heap)
            if previous is None and len(live) >= top_k:
                if not heap[0] < candidate:
                    continue
                removed = heapq.heappop(heap)
                del live[removed.source]
            live[source] = candidate
            heapq.heappush(heap, candidate)
            # Bound lazy-deletion storage even when one edge is updated often.
            if len(heap) > 2 * top_k:
                heap[:] = live.values()
                heapq.heapify(heap)
    rows = [(source, target, edge.weight)
            for target, (_, live) in targets.items()
            for source, edge in live.items()]
    rows.sort(key=lambda row: (-abs(row[2]), row[0], row[1]))
    return pd.DataFrame(rows, columns=COLUMNS)


def top_indices(weights, names, top_k, exclude=None):
    """Partial selection with deterministic boundary ties and no self-links."""
    candidates = np.flatnonzero(np.isfinite(weights))
    if exclude is not None:
        candidates = candidates[candidates != exclude]
    if top_k is not None and len(candidates) > top_k:
        magnitudes = np.abs(weights[candidates])
        threshold = np.partition(magnitudes, len(candidates) - top_k)[-top_k]
        stronger = candidates[magnitudes > threshold]
        tied = candidates[magnitudes == threshold]
        tied = sorted(tied, key=lambda index: str(names[index]))
        candidates = list(stronger) + tied[:top_k - len(stronger)]
    return sorted(candidates, key=lambda index: (-abs(weights[index]), str(names[index])))


def reduce_frame(edges, top_k=None):
    """Vectorized equivalent for edge tables that are already in memory."""
    frame = edges[COLUMNS].copy()
    frame = frame[frame.Gene1.notna() & frame.Gene2.notna()]
    frame['Gene1'] = frame.Gene1.astype(str)
    frame['Gene2'] = frame.Gene2.astype(str)
    frame['EdgeWeight'] = pd.to_numeric(frame.EdgeWeight, errors='coerce')
    frame = frame[(frame.Gene1 != frame.Gene2) & (frame.Gene1 != '') &
                  (frame.Gene2 != '') & np.isfinite(frame.EdgeWeight)]
    frame['_magnitude'] = frame.EdgeWeight.abs()
    frame = frame.sort_values(
        ['_magnitude', 'Gene1', 'Gene2', 'EdgeWeight'],
        ascending=[False, True, True, False], kind='stable',
    ).drop_duplicates(['Gene1', 'Gene2'])
    if top_k is not None:
        frame = frame.groupby('Gene2', sort=False, group_keys=False).head(top_k)
    return frame[COLUMNS].reset_index(drop=True)
