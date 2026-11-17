"""kNN-graph backends compared in isolation: wall time and recall against the exact graph.

Every embedding algorithm starts from a kNN graph, so this module times the search stage on its
own. Backends are registered in `backends.py`, the loop and its resumable JSONL store are in
`bench.py`, and `report/knn.py` renders `KNN.md`.
"""
