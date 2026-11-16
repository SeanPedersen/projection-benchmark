"""DBpedia entities embedded with OpenAI text-embedding-3-small (512d), 100k rows (Qdrant, HF Hub)."""

from pathlib import Path

import numpy as np

PARQUET_URL = (
    "https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-512-100K"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
EMBEDDING_COLUMN = "text-embedding-3-small-512-embedding"
DOWNLOAD_CHUNK_BYTES = 1 << 20


# niquests / pyarrow are imported inside the functions: benchmark workers import this
# module, and importing them at module level would add ~35 MB to every measured RSS.
def download(url: str, target: Path) -> None:
    import niquests

    partial = target.with_suffix(target.suffix + ".part")
    with niquests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as sink:
            for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                sink.write(chunk)
    partial.rename(target)


def embeddings_from_parquet(path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    column = pq.read_table(path, columns=[EMBEDDING_COLUMN]).column(EMBEDDING_COLUMN).combine_chunks()
    flat = column.flatten().to_numpy(zero_copy_only=False)
    X = flat.reshape(len(column), -1).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def prepare(target_dir: Path) -> None:
    parquet = target_dir / "raw.parquet"
    if not parquet.exists():
        download(PARQUET_URL, parquet)
    np.save(target_dir / "X.npy", embeddings_from_parquet(parquet))
