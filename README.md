# drbench: dimensionality reduction benchmark

Which implementation gives the best 2D projection for the least compute?
Each algorithm runs end to end on raw vectors, with the same CPU budget for every run. Each run is scored on:

- **Local structure:** trustworthiness at k = 5, 15, 30. Continuity and kNN recall@15 are reported alongside.
- **Global structure:** Spearman ρ of pairwise k-means centroid distances (k = 5, 15, 30), original vs. 2D. A medoid variant and a random-pair distance ρ are reported alongside.
- **Cost:** fit time and peak RSS.

## Quick start

```sh
uv sync                     # Python 3.12; builds the vendored ncvis (needs gcc + OpenMP)
uv run pytest
uv run drbench all          # prep → run → evaluate → report (full config: 100k × 8 algos × 3 seeds)
uv run drbench all --subsample 10000 --seeds 0 --results-dir results-smoke   # ~5 min smoke run
uv run drbench curves       # quality vs time: every algorithm over a grid of epoch budgets
```

Results go to `results/`: `runs.jsonl`, `metrics.jsonl`, `summary.csv`, `REPORT.md` and `figures/`.
The steps resume where they left off: finished runs and evaluations are skipped. Pass `--retry-failed` to rerun failures and timeouts.

## Results (dbpedia-512-100k, 8 threads, i7-8550U)

Full tables: [`results/REPORT.md`](results/REPORT.md) (library defaults, 3 seeds) and
[`results/curves/CURVES.md`](results/curves/CURVES.md) (quality vs time across epoch budgets).

| algorithm | fit time | peak memory | local (trustworthiness) | global (centroid ρ) |
|---|---|---|---|---|
| mrs-tsne-bh | 172 s | 1.0 GB | 0.986 | 0.74 |
| mrs-tsne-fft | 84 s | 1.0 GB | 0.986 | 0.71 |
| opentsne-fft | 141 s | 1.3 GB | 0.985 | 0.76 |
| ncvis-ef64 | 31 s | 0.5 GB | 0.961 | 0.63 |
| ncvis | 27 s | 0.5 GB | 0.954 | 0.64 |
| umap-learn | 61 s | 1.0 GB | 0.936 | 0.67 |
| mrs-umap | 18 s | 0.8 GB | 0.907 | 0.59 |
| pca | 0.3 s | 0.5 GB | 0.724 | 0.78 |

- t-SNE reaches its quality plateau at about half its default epochs: mrs-tsne-fft at ×½ gives 0.985 in 64 s.
- NCVis is the best choice under 30 s. Its stock HNSW search reaches only 0.85 recall@15. With query
  ef=64 (patched), recall is 0.94 and the local score rises by 0.0065 for 4 s more. A near-exact graph
  does not help beyond that.
- The UMAP implementations stop improving at 0.91–0.94 local score.

## Dataset (milestone 1)

`dbpedia-512-100k` is 100k DBpedia entities embedded with OpenAI `text-embedding-3-small` (512-d).
It comes from [Qdrant on the HF Hub](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-512-100K)
and is L2-normalised. `name@N` selects a fixed random subsample.

## Algorithms

| name | library | language |
|---|---|---|
| `pca` | scikit-learn | Python |
| `opentsne-fft` | openTSNE | Python / C++ |
| `mrs-tsne-bh`, `mrs-tsne-fft` | manifolds-rs | Rust (PyPI bindings) |
| `umap-learn` (0.6 pipeline), `umap-learn-legacy` (0.5 pipeline) | umap-learn | Python / numba |
| `mrs-umap` | manifolds-rs | Rust |
| `ncvis` | ncvis | C++ |
| `ncvis-ef64`: stock HNSW graph, query ef=64 (recall@15 ≈0.94 vs 0.85) | ncvis (patched) | C++ |
| `ncvis-hq`: HNSW M=16, efC=200, ef=64 (recall@15 ≈0.99) | ncvis (patched) | C++ |
| `mrs-pacmap` (registered, not in the default sweep) | manifolds-rs | Rust |

Shared hyperparameters are aligned across implementations: perplexity 30, n_neighbors 15 and min_dist 0.1. Everything else stays at library defaults.

## Two views

- **Defaults (`drbench all`).** Every library runs with its default settings, 3 seeds, scored exactly on all N. This answers "what do I get out of the box".
- **Curves (`drbench curves`).** Defaults ship very different training lengths: NCVis runs 50 epochs, UMAP 200–500, t-SNE 750–1000 iterations. So each tunable algorithm reruns with its default epochs scaled by ×⅛, ×¼, ×½, ×1 and ×2.
  - Early-exaggeration and init phases scale along with the main schedule.
  - An algorithm still improving between ×1 and ×2 gets one more run at ×4.
  - Curves are plotted against wall time and CPU time, never epochs, since epochs aren't comparable across algorithms.
  - Each point is a separate full run, so the kNN setup cost shows up as where each curve starts.
  - `CURVES.md` also lists the best local score each algorithm reaches within 5–300 s of wall time.
  - It uses 1 seed. Local scores are estimated on 10k fixed anchor points; their ranks are still exact over all N.

## Measurement protocol

- **Isolation:** every (dataset, algorithm, seed) runs in a fresh worker process. The worker gets the same thread count through `OMP/OPENBLAS/MKL/NUMBA/RAYON_NUM_THREADS` and the same CPU affinity.
- **Warm-up:** an untimed warm-up on 2,000 rows runs first, so numba JIT compilation is not measured.
- **Time:** wall clock of `fit_transform` only. kNN search is included; data loading is not. CPU time (user + system over all threads) is recorded too, since libraries use the cores to different degrees.
- **Peak memory:** the kernel's VmHWM, reset after warm-up. The worker measures it, not wait4, because `ru_maxrss` inherits the parent's RSS across fork. Values include the interpreter, the data and loaded libraries; `delta_rss_mb` subtracts the pre-fit RSS.
- **Laptop throttling:** the run order is shuffled with a fixed seed, and `cooldown_s` adds a pause between runs. Time is the median over seeds.
- **Evaluation** runs as a separate step, using exact ranks at full N: chunked BLAS distances plus a numba counting kernel, with no N×N matrix.

## TODO

- Datasets: FMNIST, image and audio embedding sets.
- Higher target dimensions (32, 128), scored by recall.
- Algorithms: umap-rs (a Rust CLI is needed, since it has no Python bindings), fast-umap (GPU), PaCMAP in the default sweep, TriMap, VAE, SOM, (F)LAS, SAUCIE, SQuaD-MDS, IsoMap, DiRe-JAX.
