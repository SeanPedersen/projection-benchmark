# drbench: dimensionality reduction benchmark

Which implementation gives the best 2D projection for the least compute?
Each algorithm runs end to end on raw vectors, with the same CPU budget for every run. Each run is scored on:

- **Local structure:** trustworthiness at k = 5, 15, 30. Continuity and kNN recall@15 are reported alongside.
- **Global structure:** Spearman ρ between original and 2D distances of 1M random point pairs. Triplet accuracy is reported alongside: the share of 1M random triplets (i, j, l) where "is j closer to i than l?" has the same answer in both spaces. Both use fixed samples whose sampling noise was measured (sd 0.001 and 0.0002–0.0005 on a fixed embedding), below the spread across seeds. k-means centroid scores were dropped because refitting k-means alone moved them by 0.01–0.08 ([#2](../../issues/2)).
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

## Results (dbpedia-512-100k)

### Library defaults (8 threads, i7-8550U)

Full tables: [`results/REPORT.md`](results/REPORT.md) (library defaults, 3 seeds) and
[`results/curves/CURVES.md`](results/curves/CURVES.md) (quality vs time across epoch budgets).
Both still show the old k-means global score until they are re-evaluated (`drbench evaluate` and `drbench curves` rescore them).
The global and triplet columns below are computed on the same embeddings with the new measures.

| algorithm | fit time | peak memory | local (trustworthiness) | global (pair ρ) | triplet |
|---|---|---|---|---|---|
| mrs-tsne-bh | 172 s | 1.0 GB | 0.986 | 0.269 | 0.596 |
| mrs-tsne-fft | 84 s | 1.0 GB | 0.986 | 0.272 | 0.597 |
| opentsne-fft | 141 s | 1.3 GB | 0.985 | 0.289 | 0.602 |
| ncvis-ef64 | 31 s | 0.5 GB | 0.961 | 0.207 | 0.576 |
| ncvis | 27 s | 0.5 GB | 0.954 | 0.213 | 0.577 |
| umap-learn | 61 s* | 1.0 GB | 0.936 | 0.274 | 0.596 |
| mrs-umap | 18 s | 0.8 GB | 0.907 | 0.259 | 0.591 |
| pca | 0.3 s | 0.5 GB | 0.724 | 0.345 | 0.629 |

\* Includes about 40 s of numba JIT from a too-short warm-up; about 23 s without it ([#1](../../issues/1)).

- t-SNE reaches its quality plateau at about half its default epochs: mrs-tsne-fft at ×½ gives 0.985 in 64 s.
- NCVis is the best choice under 30 s. Its stock HNSW search reaches only 0.85 recall@15. With query
  ef=64 (patched), recall is 0.94 and the local score rises by 0.0065 for 4 s more. A near-exact graph
  does not help beyond that.
- The UMAP implementations stop improving at 0.91–0.94 local score.
- PCA keeps global distances best (pair ρ 0.345) but has by far the worst local structure. Among the
  nonlinear methods openTSNE scores highest globally; NCVis lowest.

### kNN backend: shared RINND graph (36 threads, i9-7980XE)

Every library that accepts a precomputed kNN graph also runs as a `-rinnd-fast` variant on a
shared [RINND](https://pypi.org/project/rinnd/) graph (see [Measurement protocol](#measurement-protocol)).
3 seeds, local ± seed sd. Times are not comparable to the 8-thread table above.

| algorithm | kNN graph | fit time | peak memory | local | global (pair ρ) | triplet |
|---|---|---|---|---|---|---|
| ncvis | HNSW (stock) | 7.0 s | 540 MB | 0.9548 ± 0.0012 | 0.226 | 0.582 |
| **ncvis-rinnd-fast** | RINND k=15 | **3.9 s** | **426 MB** | **0.9621** ± 0.0006 | 0.229 | 0.582 |
| mrs-umap | NN-Descent | 4.8 s | 1029 MB | 0.9060 ± 0.0018 | 0.241 | 0.585 |
| **mrs-umap-rinnd-fast** | RINND k=15 | **3.1 s** | **399 MB** | **0.9096** ± 0.0002 | **0.263** | **0.592** |
| umap-learn | PyNNDescent | 7.0 s | 992 MB | 0.9357 ± 0.0007 | 0.273 | 0.595 |
| **umap-learn-rinnd-fast** | RINND k=15 | 7.0 s | **852 MB** | **0.9375** ± 0.0008 | 0.277 | 0.596 |
| mrs-tsne-fft | IVF | 26.5 s | 1170 MB | **0.9855** ± 0.0001 | 0.271 | 0.597 |
| mrs-tsne-fft-rinnd-fast | RINND k=90 (fast) | **14.8 s** | 1210 MB | 0.9842 ± 0.0000 | 0.276 | 0.598 |
| opentsne-fft | Annoy | 98.5 s | 1437 MB | **0.9851** ± 0.0001 | **0.292** | **0.603** |
| opentsne-fft-rinnd-fast | RINND k=90 (fast) | **79.4 s** | **1040 MB** | 0.9839 ± 0.0001 | 0.280 | 0.599 |
| mrs-pacmap | IVF | 37.4 s | 2095 MB | **0.9135** ± 0.0014 | **0.262** | **0.593** |
| mrs-pacmap-rinnd-fast | RINND k=60 (fast) | **22.4 s** | 2178 MB | 0.9052 ± 0.0004 | 0.251 | 0.591 |

- **NCVis and both UMAPs: use RINND.** As fast or faster, less memory, and a better local score.
  NCVis gains most (+0.007, 1.8× faster) because its stock HNSW graph is the least accurate.
- **t-SNE: a speed trade.** 1.25–1.8× faster for about −0.0013 local; openTSNE also loses some global
  structure. A more accurate graph did not help t-SNE either (see [#1](../../issues/1)).
- **PaCMAP: keep IVF.** The fast schedule's k=60 graph (recall 0.936) costs 0.008 local and 0.011
  global. RINND's default schedule at k=60 (recall 0.991, +1.5 s) is untested.
- More kNN detail in [#1](../../issues/1): isolated backend timings and recall, manifolds-rs's own
  NN-Descent (slow, about 3.4 GB at 36 threads), and why PCA + NN-Descent + rerank does not pay off at 512-d.

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
| `mrs-pacmap` | manifolds-rs | Rust |
| kNN-backend variants: `umap-learn-rinnd`, `opentsne-fft-nndescent`/`-exact`, `mrs-tsne-fft-nndescent`/`-hnsw` | as above | |
| `-rinnd-fast` variants of `opentsne-fft`, `mrs-tsne-fft`, `umap-learn`, `mrs-umap`, `mrs-pacmap`, `ncvis` (shared RINND graph; NCVis via a patch) | as above + RINND (Rust) | |

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
- **Warm-up:** an untimed warm-up on 5,000 rows runs first, so numba JIT compilation is not measured. It must exceed 4,096 rows: below that umap-learn uses exact distances and never compiles its PyNNDescent path.
- **Time:** wall clock of `fit_transform` only. kNN search is included; data loading is not. CPU time (user + system over all threads) is recorded too, since libraries use the cores to different degrees.
- **Shared kNN graphs:** the `-rinnd-fast` variants build one RINND graph per (dataset, k) and reuse it across algorithms and seeds. A run that loads a stored graph is charged its build time (added to fit time, also recorded as `shared_knn_s`) and its extra peak memory, so it reports the same cost as building it. CPU time of the build is not charged.
- **Peak memory:** the kernel's VmHWM, reset after warm-up. The worker measures it, not wait4, because `ru_maxrss` inherits the parent's RSS across fork. Values include the interpreter, the data and loaded libraries; `delta_rss_mb` subtracts the pre-fit RSS.
- **Laptop throttling:** the run order is shuffled with a fixed seed, and `cooldown_s` adds a pause between runs. Time is the median over seeds.
- **Evaluation** runs as a separate step, using exact ranks at full N: chunked BLAS distances plus a numba counting kernel, with no N×N matrix. Scores carry a metrics version; when a definition changes, `drbench evaluate` rescores the stored embeddings.

## TODO

- Datasets: FMNIST, image and audio embedding sets.
- Higher target dimensions (32, 128), scored by recall.
- Re-evaluate `results/` and `results/curves/` with the new global measures.
- PaCMAP with RINND's default schedule at k=60.
- Algorithms: umap-rs (a Rust CLI is needed, since it has no Python bindings), fast-umap (GPU), TriMap, VAE, SOM, (F)LAS, SAUCIE, SQuaD-MDS, IsoMap, DiRe-JAX.
