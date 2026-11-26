"""Tests for overlap-aware multi-label clustering metrics."""

from hytri.evaluation.multilabel import (
    compute_pairwise_multilabel_metrics,
    get_multilabel_ground_truth_sets,
    same_multilabel_gt,
)


def test_same_multilabel_gt_overlap():
    assert same_multilabel_gt({"A", "B"}, {"B", "C"})
    assert not same_multilabel_gt({"A"}, {"B"})


def test_pairwise_perfect_clustering():
    true_sets = [{"A"}, {"A"}, {"B"}, {"B"}]
    pred = [0, 0, 1, 1]
    m = compute_pairwise_multilabel_metrics(true_sets, pred)
    assert m["pairwise_f1"] == 1.0


def test_multilabel_overlap_ground_truth():
    failures = [
        {"ground_truth_bugs": ["Chart-3", "Chart-4"]},
        {"ground_truth_bugs": ["Chart-4"]},
    ]
    sets = get_multilabel_ground_truth_sets(failures)
    assert len(sets[0]) == 2
    assert same_multilabel_gt(sets[0], sets[1])
