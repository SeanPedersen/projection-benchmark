"""Runs recorded under an older epoch schedule are rerun, and their old scores are ignored."""

from pathlib import Path

from drbench.config import load_config
from drbench.metrics import METRICS_VERSION
from drbench.runner.evaluate import current_scores
from drbench.runner.orchestrator import pending_runs
from drbench.runner.records import RunKey, append_jsonl, metrics_path, runs_path

CONFIG_PATH = Path(__file__).parent.parent / "benchmark.toml"
OLD_EPOCHS, CURRENT_EPOCHS = 1000, 750


def test_stale_epochs_rerun_and_rescored(tmp_path):
    config = load_config(CONFIG_PATH).with_run(results_dir=tmp_path)
    stale, current = RunKey("d", "mrs-tsne-fft", 0), RunKey("d", "mrs-tsne-fft", 1)
    for key, epochs in ((stale, OLD_EPOCHS), (current, CURRENT_EPOCHS)):
        append_jsonl(runs_path(tmp_path), key.as_dict() | {"status": "ok", "epochs": epochs})
        append_jsonl(metrics_path(tmp_path), key.as_dict() | {
            "metrics_version": METRICS_VERSION, "epochs": epochs, "local": 0.9})
    assert pending_runs([stale, current], config, retry_failed=False) == [stale]

    # The rerun replaces the run record; the score of the 1000-epoch run no longer counts.
    append_jsonl(runs_path(tmp_path), stale.as_dict() | {"status": "ok", "epochs": CURRENT_EPOCHS})
    assert set(current_scores(tmp_path)) == {current}
