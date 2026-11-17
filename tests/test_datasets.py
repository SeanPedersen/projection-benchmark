"""Dataset specs: `--subsample` overrides the size, otherwise a spec's own `@N` is kept."""

from drbench.datasets import with_subsample


def test_with_subsample_keeps_explicit_size():
    assert with_subsample("dbpedia-512-100k@10000", None) == "dbpedia-512-100k@10000"
    assert with_subsample("dbpedia-512-100k", None) == "dbpedia-512-100k"
    assert with_subsample("dbpedia-512-100k@10000", 3000) == "dbpedia-512-100k@3000"
