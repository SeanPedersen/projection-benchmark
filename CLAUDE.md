# CLAUDE.md

Benchmark of dimensionality-reduction implementations: quality (local/global) vs time and peak memory.
It is all Python (3.12, uv). Rust and C++ libraries come in through their Python bindings.

## Commands
- `uv sync` and `uv run pytest` (41 tests; metric tests compare against sklearn to 1e-12)
- `uv run ruff check src tests`
- `uv run drbench {prep,run,evaluate,report,all,curves,knn} [--subsample N --seeds 0,1 --algos a,b --results-dir DIR --timeout S --threads T]`
- Config: `benchmark.toml`. CLI flags override `[run]`. `[curves]` configures the budget sweep, which writes to `results/curves/`. `[knn]` configures `drbench knn` (`--backends`, `--ks`, `--repeats` override), which writes `results/knn/knn.jsonl` and `KNN.md`.

## Layout (src/drbench)
- `cli.py`: thin dispatcher. `config.py` turns the TOML into frozen dataclasses.
- `datasets/`: registry. A spec is `name` or `name@N` (subsample). `prepare` writes `data/<spec>/X.npy`. `dbpedia-pca32-100k` is derived from `dbpedia-512-100k` by PCA (`derived.py`).
- `algorithms/`: registry of `Algorithm(fit_transform(X, seed, n_threads, budget), default_epochs)`.
  - `budget` multiplies each library's default epochs; t-SNE early exaggeration and NCVis init epochs scale with it.
  - Imports are lazy, so a worker only loads the library it measures.
  - Shared hyperparameters and `scaled_epochs` live in `params.py`.
  - Variants that differ only in the kNN backend: `umap-learn` (PyNNDescent) vs `umap-learn-rinnd`; `opentsne-fft` (Annoy) vs `-nndescent` / `-exact`; `mrs-tsne-fft` (IVF) vs `-nndescent` / `-hnsw`.
- `runner/`:
  - `orchestrator.py` spawns one `worker.py` process per run and handles timeouts and resume.
  - `memory.py` reads VmHWM.
  - `records.py` defines `RunKey(dataset, algorithm, seed, budget)`, the paths and the JSONL stores.
  - `curves.py` plans the budget grid and the ×4 extension.
  - `evaluate.py` scores runs.
- `knn/`: kNN search alone. `backends.py` is the registry `(X, k, n_threads) -> indices`; `bench.py` times each backend after an untimed warm-up and scores recall against the exact graph (cached in `data/<spec>/knn_exact.npy`). Resumable.
- `metrics/`:
  - `neighbors.py` holds the exact rank kernels (BLAS chunks for high-d, a fused numba kernel for low-d).
  - `local.py` has trustworthiness, continuity and recall.
  - `global_geometry.py` has the centroid, medoid and pair Spearman scores.
  - `__init__.py` builds the cached per-dataset `reference.npz` and runs `evaluate()`. With `anchor_rows` set, local scores are estimated on that many fixed anchor points (ranks stay exact over all N).
- `report/`: `tables.py` aggregates; `plots.py` draws the Pareto charts, curves and thumbnails (dataviz palette); `__init__.py` writes `REPORT.md`; `curves.py` writes `CURVES.md`; `knn.py` writes `KNN.md`.

## Gotchas
- `ncvis` is vendored in `vendor/ncvis`: the 1.5.13 sdist from 2023, patched with a `<size_t*>` cast so it cythonizes against NumPy 2. A second patch adds `ef_search` (HNSW query ef, 0 = stock), used by the `ncvis-ef64` and `ncvis-hq` variants. With stock settings, recall@15 is only ~0.85 on 512-d data. After editing it, rebuild with `uv sync --reinstall-package ncvis`.
- `umap-learn` must use `random_state=None`, since a fixed seed forces it single-threaded. The adapter seeds numpy's global RNG instead.
- `umap-learn` pins `nearest_neighbor_backend` per variant (0.6 would otherwise pick RINND whenever it is installed). umap-learn uses exact distances below 4096 rows, so `warmup_rows` must exceed that or PyNNDescent's numba JIT (about 40 s on 100k x 512) lands in the timed run. Measured: `umap-learn` 66 s with 2000 warm-up rows, 23 s with 5000; RINND is compiled Rust and takes 23 s either way. Fit times of `umap-learn` and `umap-learn-legacy` recorded in `results/` before this fix include JIT.
- Never measure memory with `ru_maxrss` from wait4: it includes the parent's RSS at fork. Workers must not import heavy modules they don't need (pyarrow and niquests are imported lazily in `datasets/dbpedia.py`).
- manifolds-rs defaults to `min_dist=0.5`; the adapter sets 0.1 to match umap-learn.
- Never run timed sweeps concurrently (including evaluation). They share the CPU and distort the timings.
- `data/`, `results/` and `results-*/` are gitignored.
