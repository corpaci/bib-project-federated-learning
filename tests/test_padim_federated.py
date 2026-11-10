"""Federated PaDiM reproduces pooled PaDiM exactly (not approximately).

The claim the whole Stage 2 extension rests on, tested with no network and no
SuperLink: see docs/federated_padim.md.

Random noise is a legitimate input here - the identity being tested is algebraic
and holds for any images, so toy arrays keep the test fast.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from padim import PaDiM  # noqa: E402
from padim_federated import aggregate_statistics, choose_channels  # noqa: E402

N_DIMS = 20  # small: keeps n > n_dims with few images, and the test quick


def _toy(n, seed, size=64):
    """n random images, (n, 3, size, size) float32 in [0, 1]."""
    return np.random.default_rng(seed).random((n, 3, size, size), dtype=np.float32)


class _BatchInvariantPaDiM(PaDiM):
    """PaDiM with the ResNet18 replaced by an elementwise embedding.

    The real backbone runs in float32 and its output for one image depends
    slightly on which other images share the batch (~1e-6 relative, and the exact
    amount differs between CPUs). That noise is not part of the aggregation, so
    the tests of the ALGEBRA use this stand-in: pooling and per-channel scaling
    only, which give the same value for an image in any batch on any machine.
    """

    def _embed(self, x):
        import torch
        import torch.nn.functional as F
        pooled = F.avg_pool2d(x, 4)                           # (B, 3, h, w)
        j = torch.arange(448)
        w = (0.5 + (j % 7) / 7.0).view(1, -1, 1, 1)
        b = ((j % 5) / 5.0).view(1, -1, 1, 1)
        return pooled[:, j % 3] * w + b * pooled[:, (j + 1) % 3] ** 2


def _stats(parts, idx, cls=PaDiM):
    return [cls(n_dims=N_DIMS, seed=1).local_statistics(p, idx=idx) for p in parts]


def test_federated_equals_pooled():
    """The core identity: summed statistics == centralised fit on pooled images."""
    # Uneven client sizes on purpose: equal sizes would hide a weighting bug.
    parts = [_toy(40, 0), _toy(25, 1), _toy(60, 2)]
    idx = choose_channels(seed=1, n_dims=N_DIMS)

    # Batch-invariant embedding: both paths then accumulate in float64 over the
    # SAME float32 embeddings, and reordering a sum of float64s is exact to
    # rounding, so 1e-12 is the honest bar.
    central = _BatchInvariantPaDiM(n_dims=N_DIMS, seed=1)
    central.fit(np.concatenate(parts), idx=idx)

    agg = aggregate_statistics(_stats(parts, idx, _BatchInvariantPaDiM))

    assert agg["n"] == sum(len(p) for p in parts)
    assert agg["grid"] == central.grid
    np.testing.assert_allclose(agg["mean"], central.mean, rtol=1e-12, atol=1e-12)
    # inv_cov goes through a matrix inversion, which amplifies error by the
    # condition number. Needing looser than 1e-4 here would mean a real bug
    # (a missing .double(), or the n/(n-1) rescale).
    np.testing.assert_allclose(agg["inv_cov"], central.inv_cov, rtol=1e-8, atol=1e-8)


def test_federated_matches_pooled_with_real_backbone():
    """Same identity through the real ResNet18, at float32 precision.

    Clients and the pooled fit cut the images into different batches, so the
    backbone's float32 outputs differ slightly (see _BatchInvariantPaDiM). The
    model statistics must still agree far more tightly than any metric can show.
    """
    parts = [_toy(40, 0), _toy(25, 1), _toy(60, 2)]
    idx = choose_channels(seed=1, n_dims=N_DIMS)
    central = PaDiM(n_dims=N_DIMS, seed=1)
    central.fit(np.concatenate(parts), idx=idx)
    agg = aggregate_statistics(_stats(parts, idx))
    np.testing.assert_allclose(agg["mean"], central.mean, rtol=1e-3, atol=1e-5)


def test_scores_match_end_to_end():
    """The numbers that actually get reported: image-level anomaly scores."""
    parts = [_toy(40, 0), _toy(30, 1)]
    test_x = _toy(8, 99)
    idx = choose_channels(seed=1, n_dims=N_DIMS)

    central = PaDiM(n_dims=N_DIMS, seed=1)
    central.fit(np.concatenate(parts), idx=idx)
    expected, _ = central.score(test_x, return_maps=False)

    agg = aggregate_statistics(_stats(parts, idx))
    fed = PaDiM(n_dims=N_DIMS, seed=1)
    fed.load_statistics(agg["mean"], agg["inv_cov"], idx, agg["grid"])
    got, _ = fed.score(test_x, return_maps=False)

    # rtol 1e-3: the two fits batch the images differently, so the float32
    # backbone noise (not the aggregation) sets the achievable agreement.
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_single_client_is_centralised():
    """One client must reduce to the centralised case. The first assertion to write."""
    x = _toy(40, 7)
    idx = choose_channels(seed=1, n_dims=N_DIMS)
    central = PaDiM(n_dims=N_DIMS, seed=1)
    central.fit(x, idx=idx)
    agg = aggregate_statistics(_stats([x], idx))
    np.testing.assert_allclose(agg["mean"], central.mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(agg["inv_cov"], central.inv_cov, rtol=1e-8, atol=1e-8)


def test_bessel_correction_is_not_cosmetic():
    """Dropping the n/(n-1) rescale must visibly break exactness.

    Guards the subtlest line in the aggregator: np.cov uses ddof=1 while the
    moment formula gives the /n version. A regression here is otherwise silent.
    """
    parts = [_toy(30, 3), _toy(30, 4)]
    idx = choose_channels(seed=1, n_dims=N_DIMS)
    central = PaDiM(n_dims=N_DIMS, seed=1)
    central.fit(np.concatenate(parts), idx=idx)

    stats = _stats(parts, idx)
    n = sum(s["n"] for s in stats)
    S = sum(s["S"] for s in stats)
    Q = sum(s["Q"] for s in stats)
    mean = S / n
    cov_uncorrected = Q / n - np.einsum("pd,pe->pde", mean, mean)   # no n/(n-1)
    cov_correct = cov_uncorrected * (n / (n - 1))

    # The correct one matches np.cov; the uncorrected one is (n-1)/n too small.
    eye = np.eye(N_DIMS)
    inv_wrong = np.linalg.inv(cov_uncorrected[0] + 0.01 * eye)
    inv_right = np.linalg.inv(cov_correct[0] + 0.01 * eye)
    np.testing.assert_allclose(inv_right, central.inv_cov[0], rtol=1e-6, atol=1e-6)
    assert not np.allclose(inv_wrong, central.inv_cov[0], rtol=1e-6, atol=1e-6)


def test_channel_subset_is_seed_only():
    """choose_channels depends on the seed alone (so it leaks no data) and
    matches what the centralised model would have drawn for that seed."""
    a = choose_channels(seed=1, n_dims=N_DIMS)
    b = choose_channels(seed=1, n_dims=N_DIMS)
    c = choose_channels(seed=2, n_dims=N_DIMS)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)

    # Same channels as an unseeded centralised fit with that seed.
    m = PaDiM(n_dims=N_DIMS, seed=1)
    m._draw_channels(None)
    np.testing.assert_array_equal(m.idx.numpy(), a)


def test_mismatched_channels_are_not_silent_garbage():
    """Trap 1: if clients used different subsets the model is meaningless.

    Nothing can detect this from the sums alone - which is exactly why the server
    chooses idx. This test documents the failure mode rather than preventing it:
    aggregation "succeeds" and produces a DIFFERENT model from the pooled one.
    """
    parts = [_toy(40, 0), _toy(40, 1)]
    good = choose_channels(seed=1, n_dims=N_DIMS)
    central = PaDiM(n_dims=N_DIMS, seed=1)
    central.fit(np.concatenate(parts), idx=good)

    mixed = [
        PaDiM(n_dims=N_DIMS, seed=1).local_statistics(parts[0], idx=good),
        PaDiM(n_dims=N_DIMS, seed=1).local_statistics(
            parts[1], idx=choose_channels(seed=99, n_dims=N_DIMS)),
    ]
    agg = aggregate_statistics(mixed)          # no error raised: that is the point
    assert not np.allclose(agg["mean"], central.mean, rtol=1e-3, atol=1e-3)


def test_mismatched_grids_raise():
    """Different input resolutions -> incomparable patches. Fail loudly."""
    idx = choose_channels(seed=1, n_dims=N_DIMS)
    a = PaDiM(n_dims=N_DIMS, seed=1).local_statistics(_toy(30, 0, size=64), idx=idx)
    b = PaDiM(n_dims=N_DIMS, seed=1).local_statistics(_toy(30, 1, size=128), idx=idx)
    with pytest.raises(ValueError, match="patch grid"):
        aggregate_statistics([a, b])


def test_too_few_pooled_images_raises():
    """n <= d is singular even when no single client trips the fit() guard."""
    idx = choose_channels(seed=1, n_dims=N_DIMS)
    stats = _stats([_toy(8, 0), _toy(9, 1)], idx)   # 17 images, d = 20
    with pytest.raises(ValueError, match="singular"):
        aggregate_statistics(stats)


def test_empty_raises():
    with pytest.raises(ValueError, match="no client statistics"):
        aggregate_statistics([])


def test_batching_does_not_change_statistics():
    """S and Q are sums, so the batched accumulation must be exact.

    local_statistics accumulates batch by batch to bound client memory (it never
    holds the whole embedding), and this pins that the streaming is exact rather
    than merely close.

    Uses the batch-invariant embedding: with the real ResNet18 the float32
    features themselves change with the batch size (hardware-dependent, ~1e-6
    relative), which would hide whether the ACCUMULATION is exact.
    See test_backbone_batch_sensitivity_is_small.
    """
    x = _toy(37, 11)        # deliberately not a multiple of any batch size
    idx = choose_channels(seed=1, n_dims=N_DIMS)
    ref = _BatchInvariantPaDiM(n_dims=N_DIMS, seed=1).local_statistics(
        x, idx=idx, batch_size=37)
    for bs in (1, 8, 16, 32, 64):
        got = _BatchInvariantPaDiM(n_dims=N_DIMS, seed=1).local_statistics(
            x, idx=idx, batch_size=bs)
        assert got["n"] == ref["n"] and got["grid"] == ref["grid"]
        # Sums of the same float64 values, grouped differently: equal up to
        # float64 rounding (a few ULP).
        np.testing.assert_allclose(got["S"], ref["S"], rtol=1e-13, atol=0)
        np.testing.assert_allclose(got["Q"], ref["Q"], rtol=1e-13, atol=1e-13)


def test_backbone_batch_sensitivity_is_small():
    """The real backbone may give slightly different float32 features per batch size.

    How much depends on the CPU and the maths library (0 on some machines, ~1e-6
    on others). It must stay tiny: this bounds how far a federated run can sit
    from the pooled one for reasons that have nothing to do with the aggregation.
    """
    import torch
    m = PaDiM(n_dims=N_DIMS, seed=1)
    m._draw_channels(None)
    x = torch.from_numpy(_toy(8, 11))
    with torch.no_grad():
        whole = m._embed(x)
        one_by_one = torch.cat([m._embed(x[i:i + 1]) for i in range(len(x))])
    diff = (whole - one_by_one).abs().max().item()
    assert diff < 1e-3, f"unexpected backbone batch sensitivity: {diff}"


def test_no_images_raises():
    idx = choose_channels(seed=1, n_dims=N_DIMS)
    with pytest.raises(ValueError, match="no training images"):
        PaDiM(n_dims=N_DIMS, seed=1).local_statistics(_toy(0, 0), idx=idx)
