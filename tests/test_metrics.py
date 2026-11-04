from ctr.metrics import all_support, bootstrap_ci, paired_permutation_test, support_recall


def test_support_recall():
    assert support_recall({"a", "b", "x"}, ["a", "b", "c"]) == 2 / 3


def test_all_support():
    assert all_support({"a", "b", "c"}, ["a", "b"]) == 1.0
    assert all_support({"a"}, ["a", "b"]) == 0.0


def test_bootstrap_brackets_mean():
    mean, lo, hi = bootstrap_ci([0.0, 0.5, 1.0, 1.0], n_resamples=500, seed=1)
    assert lo <= mean <= hi


def test_permutation_identical_insignificant():
    a = [1.0, 0.0, 1.0, 0.0]
    assert paired_permutation_test(a, a, n_resamples=500) > 0.5


def test_permutation_clear_difference_significant():
    assert paired_permutation_test([1.0] * 20, [0.0] * 20, n_resamples=2000) < 0.05
