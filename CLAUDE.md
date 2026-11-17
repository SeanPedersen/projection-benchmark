# CLAUDE.md

Benchmark of dimensionality-reduction implementations: quality (local/global) vs time and peak memory.
It is all Python (3.12, uv). Rust and C++ libraries come in through their Python bindings.

## Commands
- Run tests and every sweep on hartmut, not locally: `uv sync` and `uv run pytest` (67 tests; metric tests compare against sklearn to 1e-12)
- `uv run ruff check src tests`
- `uv run drbench {prep,run,evaluate,report,all,curves,knn} [--subsample N --seeds 0,1 --algos a,b --results-dir DIR --timeout S --threads T]`
- Config: `benchmark.toml`. CLI flags override `[run]`. `[curves]` configures the budget sweep, which writes to `results/curves/`. `[knn]` configures `drbench knn` (`--backends`, `--ks`, `--repeats` override), which writes `results/knn/knn.jsonl` and `KNN.md`.

## Layout (src/drbench)
- `cli.py`: thin dispatcher. `config.py` turns the TOML into frozen dataclasses.
- `datasets/`: registry. A spec is `name` or `name@N` (subsample, seeded; large samples are nested permutation prefixes, so `jinaclip-512-1m@100000` ⊂ `@512000`, but `@10000` is an independent draw). `prepare` writes `data/<spec>/X.npy`. `dbpedia-pca32-100k` is derived from `dbpedia-512-100k` by PCA (`derived.py`). `jinaclip-512-1m` (`local_file.py`) symlinks an existing file on hartmut (`DRBENCH_JINACLIP_1M` overrides the path); never modify the source under /mnt/storage.
- `algorithms/`: registry of `Algorithm(fit_transform(X, seed, n_threads, budget), default_epochs)`.
  - `budget` multiplies each library's default epochs; t-SNE early exaggeration and NCVis init epochs scale with it.
  - Imports are lazy, so a worker only loads the library it measures.
  - Shared hyperparameters and `scaled_epochs` live in `params.py`, including the t-SNE schedule both libraries run (`tsne_schedule`: 750 iterations, the first 250 with early exaggeration 12; manifolds-rs would default to 1000). The learning rate still differs: openTSNE uses N/exaggeration per phase, manifolds-rs max(N/12, 200).
  - Variants that differ only in the kNN backend: `umap-learn` (PyNNDescent) vs `umap-learn-rinnd`; `opentsne-fft` (Annoy) vs `-nndescent` / `-exact`; `mrs-tsne-fft` (IVF) vs `-nndescent` / `-hnsw`.
  - `-rinnd-fast` variants (`opentsne-fft`, `mrs-tsne-fft`, `mrs-tsne-bh`, `mrs-tsne-qd` (qdtsne quick-and-dirty Barnes-Hut, depth-capped tree), `umap-learn`, `mrs-umap`, `mrs-pacmap`, `ncvis`) take a precomputed RINND graph from `knn_graph.py`: schedule per k (k=15 defaults, k=60/90 `max_candidates=20, n_iters=4`). Graphs are cached in `data/<spec>/graphs/` and shared across algorithms and seeds; a run that loads one is charged its stored build time (added to `fit_s`, also in `shared_knn_s`) and extra peak RSS. CPU time is not charged. Every run that uses a graph records its build time as `knn_s` (None otherwise), so layout time = `fit_s - knn_s`. The fast schedules were tuned at 100K; at 512K (jinaclip) recall drops to 0.87 (k=15, 60) and 0.82 (k=90).
  - `-rinnd-default` variants of the same eight use RINND's own parameters (`knn_graph.DEFAULT`); at k=15 that equals the fast schedule, so both share one cached graph file. Adapters take `graph: str | None` (None = the library's own kNN). The scaling run (`drbench scaling`) uses the default variants.
  - Graph layouts differ: manifolds-rs and openTSNE take it as is (euclidean, no self); umap-learn needs the point itself as first neighbour (`with_self`); NCVis gets squared distances.
- `runner/`:
  - `orchestrator.py` spawns one `worker.py` process per run and handles timeouts and resume. A run whose recorded `epochs` no longer match the schedule is rerun; its old score is then ignored (`evaluate.current_scores`).
  - `memory.py` reads VmHWM.
  - `records.py` defines `RunKey(dataset, algorithm, seed, budget)`, the paths and the JSONL stores.
  - `curves.py` plans the budget grid and the ×4 extension.
  - `evaluate.py` scores runs.
- `knn/`: kNN search alone. `backends.py` is the registry `(X, k, n_threads) -> indices`; `bench.py` times each backend after an untimed warm-up and scores recall against the exact graph (cached in `data/<spec>/knn_exact.npy`). Resumable.
- `metrics/`:
  - `neighbors.py` holds the exact rank kernels (BLAS chunks for high-d, a fused numba kernel for low-d).
  - `local.py` has trustworthiness, continuity and recall.
  - `global_geometry.py` has the global scores: `global` = random-pair Spearman ρ (1M pairs) and `triplet_acc` (1M triplets). Both have measured sampling sd ≤ 0.001, below the seed spread. k-means scores were dropped as unstable (issue #2); k-means labels only colour the thumbnails.
  - `__init__.py` builds the cached per-dataset `reference.npz` (missing or resized parts are recomputed on load) and runs `evaluate()`. `METRICS_VERSION` is stored with each score; `evaluate` rescores records from older versions and reports only use current ones. Above `[metrics] anchors_above_n` points, local scores are estimated on `anchor_rows` fixed anchor points (ranks stay exact over all N) and the reference kNN is computed for the anchors only (`knn_rows` in `reference.npz`); a cached full-N kNN serves any anchor set. Curves always use anchors.
- `report/`: `tables.py` aggregates; `plots.py` draws the Pareto charts, curves and thumbnails (dataviz palette); `__init__.py` writes `REPORT.md`; `curves.py` writes `CURVES.md`; `knn.py` writes `KNN.md`; `scaling.py` writes `SCALING.md` for one dataset at several `@N` sizes (`drbench scaling --datasets a@N1,a@N2 --algos ...`): a local-vs-global map per N and a table per N ranked by `combined` (mean of min-max scaled local and global across the methods at that N; triplet is left out on purpose).
- README: keep it concise and about projection quality (local + global together); cost is a side column. Detailed tables live in `results/**.md`.

## Gotchas
- `ncvis` is vendored in `vendor/ncvis`: the 1.5.13 sdist from 2023, patched with a `<size_t*>` cast so it cythonizes against NumPy 2. A second patch adds `ef_search` (HNSW query ef, 0 = stock), used by the `ncvis-ef64` and `ncvis-hq` variants. A third adds `fit_transform(X, knn_indices=, knn_distances=)` (C++ `fit_transform_knn`), which skips HNSW and embeds a given graph (`ncvis-rinnd-fast`). With stock settings, recall@15 is only ~0.85 on 512-d data. After editing it, rebuild with `uv sync --reinstall-package ncvis`.
- `umap-learn` must use `random_state=None`, since a fixed seed forces it single-threaded. The adapter seeds numpy's global RNG instead.
- `umap-learn` pins `nearest_neighbor_backend` per variant (0.6 would otherwise pick RINND whenever it is installed). umap-learn uses exact distances below 4096 rows, so `warmup_rows` must exceed that or PyNNDescent's numba JIT (about 40 s on 100k x 512) lands in the timed run. Measured: `umap-learn` 66 s with 2000 warm-up rows, 23 s with 5000; RINND is compiled Rust and takes 23 s either way. Fit times of `umap-learn` and `umap-learn-legacy` recorded in `results/` before this fix include JIT.
- Never measure memory with `ru_maxrss` from wait4: it includes the parent's RSS at fork. Workers must not import heavy modules they don't need (pyarrow and niquests are imported lazily in `datasets/dbpedia.py`).
- manifolds-rs defaults to `min_dist=0.5`; the adapter sets 0.1 to match umap-learn.
- Never run timed sweeps concurrently (including evaluation). They share the CPU and distort the timings.
- `data/`, `results/` and `results-*/` are gitignored, except the tracked summaries in `results/`, `results/curves/` and `results/scaling/` (README figures link there).
- The scaling run (2026-11-17, hartmut, `results-scaling/` there and locally) is summarised in the README; `results-scaling-fast/` holds the aborted fast-schedule attempt.
- Never `pkill -f` over ssh: the pattern matches the ssh command itself. Kill by PID.
