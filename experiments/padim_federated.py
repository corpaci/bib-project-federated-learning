"""Exact aggregation of PaDiM across clients.

See docs/federated_padim.md for the derivation.

PaDiM's per-patch mean and covariance are sufficient statistics, so summing the
clients' partial sums reproduces the pooled model EXACTLY - not approximately, the
way FedAvg approximates centralised SGD. One round, no iteration, no gradients.

Deliberately Flower-free: the maths lives here so it can be tested with pytest and
no SuperLink (tests/test_padim_federated.py).
"""

from __future__ import annotations

import numpy as np

from padim import COV_EPS, EMBED_CHANNELS


def choose_channels(seed: int, n_dims: int,
                    n_channels: int = EMBED_CHANNELS) -> np.ndarray:
    """The channel subset, drawn ONCE on the server and broadcast to every client.

    Every client must use the same subset: the aggregated covariance mixes channel
    i of one client with channel i of another, so disagreeing subsets produce a
    meaningless model and - worse - no error at all, just AUROC ~0.5.

    Depends only on the seed, so it leaks nothing about anybody's data. Matches the
    draw in PaDiM._draw_channels, so a federated run with the same seed uses the
    same channels as the centralised baseline.
    """
    return np.random.default_rng(seed).choice(n_channels, size=n_dims, replace=False)


def aggregate_statistics(stats: list[dict], cov_eps: float = COV_EPS) -> dict:
    """Sum clients' sufficient statistics into the pooled PaDiM model.

    stats: one dict per client, as returned by PaDiM.local_statistics().
    Returns {"mean": (p, d), "inv_cov": (p, d, d), "grid": (h, w), "n": int}.

    No weighting argument, because there is nothing to weight: a client with 500
    images contributes a proportionally larger S and Q automatically and exactly.
    Contrast fl_app/aitdm_fl/client_app.py, where FedAvg trusts a self-reported
    "num-examples" count.
    """
    if not stats:
        raise ValueError("no client statistics to aggregate")

    grids = {tuple(s["grid"]) for s in stats}
    if len(grids) != 1:
        raise ValueError(
            f"clients disagree on the patch grid: {sorted(grids)}. They are using "
            "different input resolutions, so their patches are not comparable.")
    grid = grids.pop()

    n = sum(int(s["n"]) for s in stats)
    d = stats[0]["S"].shape[1]
    if n <= d:
        raise ValueError(
            f"{n} training images in total for n_dims={d}: the pooled per-patch "
            "covariance is singular. Use more images or a smaller n_dims.")

    # The whole federation, in two lines: sums are additive across clients, so
    # sum_i over the pooled set == sum_k sum_{i in k}. That is the entire proof.
    S = sum(s["S"].astype(np.float64) for s in stats)      # (p, d)
    Q = sum(s["Q"].astype(np.float64) for s in stats)      # (p, d, d)

    mean = S / n                                            # (p, d)
    # Population covariance from the second moment, then Bessel-corrected to match
    # np.cov(ddof=1) in the centralised fit(). Without the n/(n-1) the federated
    # covariance is quietly (n-1)/n too small - invisible at n=570, and invisible
    # as a bug too, which is worse. With it, the exactness test passes to ~1e-9.
    cov = Q / n - np.einsum("pd,pe->pde", mean, mean)
    cov *= n / (n - 1)

    p = mean.shape[0]
    eye = np.eye(d)
    inv_cov = np.empty((p, d, d), dtype=np.float64)
    for i in range(p):
        inv_cov[i] = np.linalg.inv(cov[i] + cov_eps * eye)

    return {"mean": mean, "inv_cov": inv_cov, "grid": grid, "n": n}
