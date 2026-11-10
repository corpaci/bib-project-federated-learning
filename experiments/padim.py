"""PaDiM (Defard et al., 2020, arXiv:2011.08785) - patch distribution modelling.

One of the six unsupervised baselines benchmarked on AutoVI by Carvalho et al.
(2024), Computers in Industry 163, 104151. Chosen because "training" is a single
forward pass plus a mean and a covariance: no optimiser, no epochs.

The embedding, the covariance inversion and the Mahalanobis distance are all
batched torch ops, so this runs on CPU or GPU with identical results (pass
device=). At 128x128 CPU is fine; a GPU earns its keep at 400x400, where the
patch grid is 100x100 = 9,604 positions instead of 32x32 = 1,024.

Trains on NORMAL images only. Use data/partitions/autovi_official (--mode official),
never the supervised re-split, or the AUROC is not comparable with the paper.

For every position in a 32x32 patch grid we fit one Gaussian over the normal
training images, then score a test image by the Mahalanobis distance of each of
its patches from the Gaussian at that same position. The maximum over the grid is
the image-level anomaly score; the grid itself is a free localisation heatmap.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # scipy arrives with scikit-learn, but do not hard-require it
    gaussian_filter = None

# ImageNet statistics: the frozen backbone was trained with these, so inputs
# must be normalised the same way. Skipping this is the usual cause of AUROC ~0.5.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Regularisation added to every covariance before inversion. The paper's value;
# without it np.linalg.inv raises LinAlgError on near-singular matrices.
COV_EPS = 0.01

# Channels in the multi-scale embedding: resnet18 layer1 + layer2 + layer3.
# Fixed by the backbone, so the channel subset can be drawn before embedding
# anything - which is what lets the server choose it for the whole federation.
EMBED_CHANNELS = 64 + 128 + 256  # = 448


def _blur(m: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of one (H, W) anomaly map."""
    if gaussian_filter is not None:
        return gaussian_filter(m, sigma=sigma)
    # Fallback with no new dependency: box blur of comparable width.
    k = max(3, int(2 * round(sigma) + 1))
    t = torch.from_numpy(m)[None, None].float()
    t = F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, stride=1)
    return t.squeeze().numpy()


class PaDiM:
    """Fit on normal images, score anything. No trainable parameters."""

    def __init__(self, n_dims: int = 100, seed: int = 1, device=None,
                 inv_chunk: int = 1024):
        self.n_dims = n_dims
        self.seed = seed
        self.device = device or torch.device("cpu")
        # Patch positions inverted per torch.linalg.inv call. 1024 is the whole
        # grid at 128x128; at 400x400 (100x100 = 9,604 positions) it chunks.
        self.inv_chunk = inv_chunk

        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)          # frozen: this is the whole point

        self.idx = None       # the random channel subset, drawn in fit()
        self.mean = None      # (n_patches, n_dims)
        self.inv_cov = None   # (n_patches, n_dims, n_dims)
        self.grid = None      # (h, w) of the patch grid

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float in [0,1] -> (B, C, h, w) multi-scale embedding."""
        if x.shape[1] == 1:                  # grayscale -> 3 channels for ResNet
            x = x.repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        m = self.backbone
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f1 = m.layer1(x)      # (B,  64, 32, 32) for 128x128 input
        f2 = m.layer2(f1)     # (B, 128, 16, 16)
        f3 = m.layer3(f2)     # (B, 256,  8,  8)
        size = f1.shape[-2:]
        return torch.cat([
            f1,
            F.interpolate(f2, size=size, mode="nearest"),
            F.interpolate(f3, size=size, mode="nearest"),
        ], dim=1)             # (B, 448, 32, 32)

    def _draw_channels(self, idx: np.ndarray | None) -> None:
        """Set self.idx: the given subset, or a fresh one drawn from self.seed.

        Drawing it from the seed is the only stochastic part of PaDiM, so it is
        the only source of seed variance. In the federated setting the server
        passes idx in, because every client MUST use the same subset (channel i
        of one client is averaged with channel i of another).
        """
        if idx is None:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(EMBED_CHANNELS, size=self.n_dims, replace=False)
        self.idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))

    @torch.no_grad()
    def fit(self, x: np.ndarray, batch_size: int = 32,
            idx: np.ndarray | None = None) -> None:
        """x: (N, 3, H, W) float32 in [0,1], NORMAL images only.

        idx: the channel subset to use. None (default) draws it from self.seed,
        as the paper does. Pass it in the federated setting, where the server
        chooses one subset for the whole federation (see experiments/padim_federated.py).
        """
        if len(x) <= self.n_dims:
            raise ValueError(
                f"{len(x)} training images for n_dims={self.n_dims}: the per-patch "
                "covariance is singular. Use more images or a smaller --n-dims.")

        self._draw_channels(idx)
        # Accumulated batch by batch into sums, so peak memory depends on
        # batch_size and not on N. The sums are the same sufficient statistics
        # local_statistics() sends to the server, and the covariance is rebuilt
        # from them with the SAME algebra as aggregate_statistics(): one formula
        # for both paths instead of two that can silently drift apart.
        #
        # float64 throughout: Q accumulates sums of squares, and the cancellation
        # in Q/n - mu mu^T can cost several significant digits in float32. It is
        # also what lets tests/test_padim_federated.py hold mean to rtol 1e-12.
        stats = self._accumulate(x, batch_size)
        n, d = stats["n"], self.n_dims
        h, w = stats["grid"]
        self.grid = (h, w)

        mean = stats["S"] / n                                   # (p, d)
        # Population second moment -> covariance, then Bessel-corrected so this
        # matches np.cov(ddof=1) and the federated path exactly. Dropping the
        # n/(n-1) makes the covariance quietly (n-1)/n too small; see
        # tests/test_padim_federated.py::test_bessel_correction_is_not_cosmetic.
        cov = stats["Q"] / n - mean[:, :, None] * mean[:, None, :]
        cov *= n / (n - 1)
        cov += COV_EPS * torch.eye(d, dtype=cov.dtype, device=cov.device)

        self.mean = mean.cpu().numpy()
        self.inv_cov = self._batched_inverse(cov).cpu().numpy()

    def _batched_inverse(self, cov: torch.Tensor) -> torch.Tensor:
        """Invert a (p, d, d) stack of covariances, chunked over p.

        torch.linalg.inv is batched, so this is one call per chunk instead of one
        per patch position. Chunking bounds peak memory: at 400x400 the grid is
        100x100 = 9,604 positions, where inverting the whole stack at once needs
        an order of magnitude more scratch space than one chunk does.
        """
        return torch.cat([torch.linalg.inv(c)
                          for c in cov.split(self.inv_chunk)])

    @torch.no_grad()
    def _accumulate(self, x: np.ndarray, batch_size: int,
                    idx: torch.Tensor | None = None) -> dict:
        """Stream x and return {"n", "S", "Q", "grid"} as float64 torch tensors.

        S = sum of embeddings, Q = sum of outer products, per patch position.
        Shared by fit() and local_statistics() so the centralised and federated
        paths accumulate through exactly the same arithmetic.
        """
        sel = (self.idx if idx is None else idx).to(self.device)
        d = len(sel)
        n_seen, grid, S, Q = 0, None, None, None
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            emb = self._embed(batch)[:, sel]
            b, _, h, w = emb.shape
            if S is None:
                grid = (h, w)
                S = torch.zeros(h * w, d, dtype=torch.float64, device=self.device)
                Q = torch.zeros(h * w, d, d, dtype=torch.float64, device=self.device)
            elif grid != (h, w):
                raise ValueError(
                    f"batch {i // batch_size} has grid {(h, w)}, expected {grid}: "
                    "the client's images are not all the same size.")
            # (b, d, h*w) -> (p, b, d): patch position first, so the per-position
            # outer products are one batched matmul.
            e = emb.reshape(b, d, h * w).permute(2, 0, 1).double()
            S += e.sum(dim=1)
            Q.baddbmm_(e.transpose(1, 2), e)        # Q += e^T e, per position
            n_seen += b
        if S is None:
            raise ValueError("no images: nothing to accumulate statistics from")
        return {"n": n_seen, "S": S, "Q": Q, "grid": grid}

    @torch.no_grad()
    def local_statistics(self, x: np.ndarray, idx: np.ndarray,
                         batch_size: int = 32) -> dict:
        """Sufficient statistics of one client's NORMAL images, for federation.

        Returns {"n": int, "S": (n_patches, d), "Q": (n_patches, d, d), "grid": (h, w)}
        where S = sum of embeddings and Q = sum of outer products, per patch
        position. Mean and covariance are reconstructible EXACTLY from these sums,
        so the server can build the same model it would have built on the pooled
        images. No image, and no per-image quantity, leaves the client.

        Deliberately does NOT invert anything: inv(mean) != mean(inv), so the
        server must sum the second moments and invert once (trap 2 of the guide).
        """
        if len(x) == 0:
            raise ValueError("no training images: nothing to compute statistics from")
        self._draw_channels(idx)                     # server-chosen, never drawn here

        # Exactly the same accumulation fit() uses, so a client's statistics and
        # the centralised ones are produced by identical arithmetic - which is what
        # makes the federated model bitwise-comparable rather than merely close.
        # Peak memory depends on batch_size, never on how many images the client
        # owns, because S and Q are sums.
        stats = self._accumulate(x, batch_size)
        # NumPy on the way out: fl_app/aitdm_fl/padim_client_app.py wraps S and Q
        # in Flower Arrays, and aggregate_statistics() sums them with numpy.
        return {"n": int(stats["n"]), "S": stats["S"].cpu().numpy(),
                "Q": stats["Q"].cpu().numpy(), "grid": stats["grid"]}

    def load_statistics(self, mean: np.ndarray, inv_cov: np.ndarray,
                        idx: np.ndarray, grid: tuple[int, int]) -> None:
        """Make this instance scorable from an aggregated model (no fit, no images).

        The server side of federation: score() only reads mean, inv_cov and idx,
        so a model assembled from clients' statistics scores exactly like a fitted one.
        """
        self.mean = np.asarray(mean)
        self.inv_cov = np.asarray(inv_cov)
        self.idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        self.grid = tuple(grid)

    @torch.no_grad()
    def score(self, x: np.ndarray, batch_size: int = 32, sigma: float = 4.0,
              return_maps: bool = True):
        """Return (image_scores (N,), anomaly_maps (N, H, W) or None)."""
        if self.mean is None:
            raise RuntimeError("call fit() before score()")
        out_size = x.shape[-1]
        idx = self.idx.to(self.device)
        # Moved to the device once, not per batch. float64 to match the fit, so
        # the quadratic form is computed at the precision the covariance was
        # inverted at.
        mean = torch.as_tensor(self.mean, dtype=torch.float64, device=self.device)
        inv_cov = torch.as_tensor(self.inv_cov, dtype=torch.float64,
                                  device=self.device)
        scores, maps = [], []

        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            emb = self._embed(batch)[:, idx]                      # (b, d, h, w)
            b, d, h, w = emb.shape
            # (b, d, h*w) -> (b, p, d): patch position second, so one einsum
            # covers every position instead of a Python loop over p. This is the
            # whole reason a GPU helps here at all.
            e = emb.reshape(b, d, h * w).transpose(1, 2).double()
            delta = e - mean                                      # (b, p, d)
            # Mahalanobis distance at every position at once. clamp before sqrt:
            # float error can make the quadratic form slightly negative and
            # produce NaN.
            dist = torch.einsum("bpi,pij,bpj->bp", delta, inv_cov, delta)
            m = dist.clamp(min=0).sqrt().reshape(b, 1, h, w).float()

            m = F.interpolate(m, size=out_size, mode="bilinear",
                              align_corners=False).squeeze(1).cpu().numpy()
            m = np.stack([_blur(mi, sigma) for mi in m])
            scores.append(m.reshape(len(m), -1).max(axis=1))
            if return_maps:
                maps.append(m.astype(np.float32))

        return (np.concatenate(scores),
                np.concatenate(maps) if return_maps else None)
