"""Run identity, result paths and the append-only JSONL stores."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_BUDGET = 1.0


@dataclass(frozen=True, order=True)
class RunKey:
    dataset: str
    algorithm: str
    seed: int
    budget: float = DEFAULT_BUDGET

    @property
    def stem(self) -> str:
        suffix = "" if self.budget == DEFAULT_BUDGET else f"-x{self.budget:g}"
        return f"seed{self.seed}{suffix}"

    def embedding_path(self, results_dir: Path) -> Path:
        return results_dir / "embeddings" / self.dataset / self.algorithm / f"{self.stem}.npy"

    def log_path(self, results_dir: Path) -> Path:
        return results_dir / "logs" / self.dataset / f"{self.algorithm}-{self.stem}.log"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def of(cls, record: dict) -> "RunKey":
        budget = float(record.get("budget", DEFAULT_BUDGET))
        return cls(record["dataset"], record["algorithm"], int(record["seed"]), budget)


def runs_path(results_dir: Path) -> Path:
    return results_dir / "runs.jsonl"


def metrics_path(results_dir: Path) -> Path:
    return results_dir / "metrics.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_by_key(records: list[dict]) -> dict[RunKey, dict]:
    """Later records win, so a retried run supersedes its failed attempt."""
    return {RunKey.of(record): record for record in records}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as sink:
        sink.write(json.dumps(record) + "\n")
