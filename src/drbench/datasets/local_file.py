"""Datasets that already exist as a `.npy` file on the benchmark machine.

`prepare` symlinks the file into the data directory instead of copying it, so the source is never
modified and costs no extra disk. The source path is a module constant, overridable per dataset by
an environment variable.

- `jinaclip-512-1m`: 999,500 × 512 float32, L2-normalised. Most likely jina-clip-v2 caption
  embeddings (the diskovec 1M benchmark corpus on hartmut); the provenance of this exact file is
  not confirmed.
"""

import os
from pathlib import Path

JINACLIP_1M_PATH = Path("/mnt/storage/diskovec-1m512-20261022/work/data/base.npy")
JINACLIP_1M_ENV = "DRBENCH_JINACLIP_1M"


def symlinked(default_path: Path, env_var: str):
    """Build a `prepare(target_dir)` that links `X.npy` to an existing float32 `.npy` file."""

    def prepare(target_dir: Path) -> None:
        source = Path(os.environ.get(env_var, default_path)).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{source} missing; set {env_var} to the .npy file")
        (target_dir / "X.npy").symlink_to(source)

    return prepare
