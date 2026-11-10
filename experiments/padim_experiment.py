"""PaDiM baseline runner: pooled and per-client, over several seeds.

The unsupervised arm of Stage 1. PaDiM is training-free (see experiments/padim.py),
so a "run" is a forward pass plus a mean and a covariance per patch position.

    python experiments/padim_experiment.py --partitions data/partitions/autovi_official
    python experiments/padim_experiment.py --partitions data/partitions/autovi_official \\
        --seeds 1 2 3 --name padim-autovi

Arms reported:
  pooled                - one PaDiM on all clients' normal training images (the
                          literal "centralised" baseline Stage 1 asks for)
  client_<i>            - one PaDiM per client, scored on its own slice of the
                          test set (what the paper does: per-category AUROC)
  central_on_client<i>  - the POOLED model scored on that same slice

The last two are the controlled comparison: identical images, and the training
set is the only thing that differs, so per_client_mean vs central_on_own_slice_mean
answers "would this site be better off training alone?". per_client_mean vs pooled
cannot answer it - it changes the evaluation set at the same time as the training
set, so a gap there might only mean one client's slice is easier.

Requires the OFFICIAL protocol partitions (normal-only train). Pointed at the
supervised re-split it raises instead of quietly discarding the labelled defects:
those images are test positives here, not rubbish.

Primary metric is AUROC, which needs no threshold and is what the AutoVI paper
reports. The thresholded numbers use the 95th percentile of the VALIDATION
normals' scores - no anomalies and no test data, so nothing leaks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from padim import PaDiM  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "fl_app"))
from aitdm_fl.task import get_device, load_npz_arrays, set_seed  # noqa: E402

# A priori false-alarm rate on known-good parts; fixed, never tuned.
NORMAL_QUANTILE = 95.0

# Every metric aggregated over seeds and printed. One list, because it drives both
# per_client_mean and summary["arms"]: two hard-coded tuples is how a metric gets
# added in one place and silently dropped from the other.
# auroc/AP are threshold-free; the rest read the threshold. tnr is the calibration
# check - the threshold IS the 95th percentile of validation normals, so a test tnr
# far from 0.95 means the validation normals do not represent the test normals and
# every thresholded number below is mis-calibrated.
METRICS = ("auroc", "average_precision", "balanced_accuracy_at_threshold",
           "tpr_at_95_tnr", "accuracy", "f1_macro", "tpr", "tnr")


def check_normal_only(y: np.ndarray, partitions: Path, what: str) -> None:
    """The protocol guarantee, enforced loudly rather than silently repaired."""
    n_anom = int((y == 1).sum())
    if n_anom:
        raise SystemExit(
            f"{n_anom} anomalies in the {what} under {partitions}. "
            "PaDiM models NORMAL appearance: training on defects teaches it that defects "
            "are normal and collapses the AUROC. You are probably pointing at the "
            "supervised partitions - use data/partitions/autovi_official (--mode official)."
        )


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray,
                        threshold: float) -> dict[str, float]:
    """AUROC and AP are threshold-free; the rest need the threshold.

    Ranking metrics need both classes present and are nan on a single-class slice.
    accuracy/f1/tpr/tnr are computed per class where defined, so a normals-only
    slice still reports an honest accuracy and tnr instead of nothing.
    """
    both_classes = len(np.unique(y_true)) == 2
    out: dict[str, float] = {}
    if not both_classes:
        # A single-class slice has no ROC. Report what is defined, flag the rest.
        out["auroc"] = float("nan")
        out["average_precision"] = float("nan")
        out["tpr_at_95_tnr"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["average_precision"] = float(average_precision_score(y_true, scores))
        # TPR at 95% TNR: the operating point the paper discusses.
        tnr_threshold = float(np.percentile(scores[y_true == 0], NORMAL_QUANTILE))
        out["tpr_at_95_tnr"] = float((scores[y_true == 1] > tnr_threshold).mean())

    y_pred = (scores > threshold).astype(np.int64)
    out["balanced_accuracy_at_threshold"] = (
        float(balanced_accuracy_score(y_true, y_pred)) if both_classes else float("nan"))
    out["accuracy"] = float((y_pred == y_true).mean())
    # zero_division=0: a single-class slice otherwise warns and returns a
    # misleading value for the absent class.
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["tpr"] = (float(y_pred[y_true == 1].mean())
                  if (y_true == 1).any() else float("nan"))
    out["tnr"] = (float(1.0 - y_pred[y_true == 0].mean())
                  if (y_true == 0).any() else float("nan"))
    out["threshold"] = float(threshold)
    out["n_images"] = int(len(y_true))
    out["n_anomalies"] = int((y_true == 1).sum())
    return out


def evaluate_global_test(model, partitions: Path, sigma: float = 4.0,
                         batch_size: int = 32) -> dict:
    """Score the global test set with a fitted or aggregated PaDiM model.

    The threshold is the 95th percentile of the clients' pooled VALIDATION
    normals - no anomalies and no test data, so nothing leaks. Used by the pooled
    arm's federated counterparts (Flower server and experiments/federated.py), so
    all of them are thresholded and scored the same way.
    """
    xs = []
    for path in sorted(partitions.glob("client_*.npz"),
                       key=lambda p: int(p.stem.split("_")[1])):
        x, y = load_npz_arrays(path, "val")
        xs.append(x[y == 0])
    if not xs:
        raise FileNotFoundError(f"no client_*.npz in {partitions}")
    val_scores, _ = model.score(np.concatenate(xs), batch_size=batch_size,
                                sigma=sigma, return_maps=False)
    threshold = float(np.percentile(val_scores, NORMAL_QUANTILE))

    x_test, y_test = load_npz_arrays(partitions / "test.npz", "test")
    scores, _ = model.score(x_test, batch_size=batch_size, sigma=sigma,
                            return_maps=False)
    return metrics_from_scores(y_test, scores, threshold)


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(arr):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    # ddof=1 (sample std), matching experiments/summarize.py, which reports the
    # supervised arms. With ddof=0 the PaDiM error bars came out sqrt(2/3) ~ 0.82
    # of the supervised ones over 3 seeds, so the two tables' +- meant different
    # things. Guarded because ddof=1 on a single seed divides by zero.
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(len(arr))}


# AutoVI is CC BY-NC-SA 4.0. Any figure containing its images carries the credit.
AUTOVI_CREDIT = "AutoVI (c) Renault Group - CC BY-NC-SA 4.0"


def _pyplot():
    """matplotlib with the Agg backend, or None on a machine without it.

    Agg before pyplot: a compute node has no display, and a missing figure must
    never fail a job.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def save_heatmaps(run_dir: Path, x: np.ndarray, maps: np.ndarray, y: np.ndarray,
                  scores: np.ndarray, per_class: int = 2,
                  pick: str = "random", seed: int = 0) -> int:
    """A few image/heatmap pairs per class. Free interpretability, costs nothing.

    pick="random" (default) samples typical images, which is what a report wants.
    pick="top" takes the highest-scoring of each class - useful for debugging,
    but misleading as an illustration, because the highest-scoring NORMALS are
    the near-misses rather than anything representative.
    """
    plt = _pyplot()
    if plt is None:
        print("    (matplotlib missing, skipping heatmaps)")
        return 0

    out = run_dir / "heatmaps"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    for label, tag in ((0, "normal"), (1, "anomaly")):
        idx = np.flatnonzero(y == label)
        if not len(idx):
            continue
        if pick == "top":
            chosen = idx[np.argsort(-scores[idx])][:per_class]
        else:
            chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for rank, i in enumerate(chosen):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
            axes[0].imshow(x[i].transpose(1, 2, 0).clip(0, 1))
            axes[0].set_title(f"{tag} (score {scores[i]:.1f})", fontsize=9)
            axes[1].imshow(x[i].transpose(1, 2, 0).clip(0, 1))
            # One colour scale across the whole figure set, so panels are
            # comparable: per-panel autoscaling makes a quiet normal map look as
            # hot as a real defect.
            axes[1].imshow(maps[i], cmap="jet", alpha=0.5,
                           vmin=float(maps.min()), vmax=float(maps.max()))
            axes[1].set_title("PaDiM anomaly map", fontsize=9)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(AUTOVI_CREDIT, fontsize=7, y=0.02)
            fig.tight_layout()
            fig.savefig(out / f"{tag}_{rank}.png", dpi=110)
            plt.close(fig)
            written += 1
    return written


def save_score_figure(path: Path, y: np.ndarray, scores: np.ndarray,
                      threshold: float, title: str) -> bool:
    """Score histogram by class + ROC curve, with the threshold drawn in.

    This is the picture behind the guide's "no AUROC is exactly 1.00 and none is
    ~0.50" check: it separates "AUROC 0.62 because the task is hard" (broad
    overlap) from "AUROC 0.62 because of a handful of outliers" (a long tail).
    """
    plt = _pyplot()
    if plt is None:
        print("    (matplotlib missing, skipping the score figure)")
        return False
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(float(scores.min()), float(scores.max()), 40)
    ax[0].hist(scores[y == 0], bins=bins, alpha=0.6, label="normal")
    ax[0].hist(scores[y == 1], bins=bins, alpha=0.6, label="anomaly")
    ax[0].axvline(threshold, color="k", ls="--", lw=1,
                  label=f"threshold {threshold:.1f}")
    ax[0].set_xlabel("image anomaly score")
    ax[0].set_ylabel("test images")
    ax[0].legend()

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, scores)
        ax[1].plot(fpr, tpr, label=f"AUROC {roc_auc_score(y, scores):.3f}")
    ax[1].plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax[1].set_xlabel("false positive rate")
    ax[1].set_ylabel("true positive rate")
    ax[1].legend()

    fig.suptitle(f"{title}\n{AUTOVI_CREDIT}", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaDiM unsupervised baseline (AutoVI official protocol).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--partitions", type=Path, required=True,
                        help="official-protocol partitions (normal-only train)")
    parser.add_argument("--n-dims", type=int, default=100,
                        help="random channel subset kept out of 448")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Gaussian blur on the anomaly map")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--name", help="experiment name (default: padim-<partitions>)")
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--heatmaps-per-class", type=int, default=4)
    parser.add_argument("--heatmap-pick", choices=["random", "top"], default="random",
                        help="random: typical images, for a report. top: the "
                             "highest-scoring of each class, for debugging.")
    parser.add_argument("--figures", action="store_true",
                        help="also write a score-histogram + ROC figure (first seed) "
                             "to outputs/figures/")
    parser.add_argument("--device", default=None,
                        help="torch device (e.g. cuda, cuda:0, cpu). Default: cuda "
                             "if available. The embedding, the batched covariance "
                             "inversion and the Mahalanobis einsum all run here.")
    parser.add_argument("--inv-chunk", type=int, default=1024,
                        help="patch positions inverted per torch.linalg.inv call. "
                             "Lower it if a large-image run runs out of memory: at "
                             "400x400 the grid is 100x100 = 9,604 positions.")
    parser.add_argument("--allow-contaminated-train", action="store_true",
                        help="Option B fallback: filter anomalies out of train with a "
                             "warning instead of raising. Not comparable with the paper.")
    args = parser.parse_args()

    partitions = args.partitions.resolve()
    name = args.name or f"padim-{partitions.name}"
    exp_dir = REPO_ROOT / "outputs" / "experiments" / name
    device = torch.device(args.device) if args.device else get_device()

    files = sorted(partitions.glob("client_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"no client_*.npz in {partitions}")

    # --- training images, per client, normal-only ---------------------------
    client_train = []
    for path in files:
        x_tr, y_tr = load_npz_arrays(path, "train")
        if args.allow_contaminated_train:
            keep = y_tr == 0
            if (~keep).sum():
                print(f"WARNING {path.name}: dropping {int((~keep).sum())} anomalies from "
                      "train (--allow-contaminated-train). NOT comparable with the paper.")
            x_tr = x_tr[keep]
        else:
            check_normal_only(y_tr, partitions, f"training split of {path.name}")
        client_train.append(x_tr)

    # --- validation sets, with labels ---------------------------------------
    # Both x and y are kept: the normals alone set the threshold, but the labelled
    # val set is what makes the local-vs-central comparison below possible at all.
    client_val = []
    for path in files:
        x_va, y_va = load_npz_arrays(path, "val")
        client_val.append((x_va, y_va))

    # --- test set, with owner ids for the per-client arm --------------------
    test = np.load(partitions / "test.npz")
    y_test = test["y_test"].reshape(-1).astype(np.int64)
    x_test = test["x_test"].astype(np.float32)
    if x_test.ndim == 4 and x_test.shape[-1] in (1, 3):
        x_test = x_test.transpose(0, 3, 1, 2)
    x_test = x_test / 255.0
    if x_test.shape[1] == 1:
        x_test = np.repeat(x_test, 3, axis=1)
    # load_npz_arrays returns only (x, y), so owner_test is read directly.
    owner = (test["owner_test"].reshape(-1).astype(np.int64)
             if "owner_test" in test.files else None)
    if owner is None:
        print(f"WARNING: no owner_test in {partitions}/test.npz - the per-client TEST-slice "
              "arm is skipped (the local-vs-central arms on each client's own validation "
              "set still run). Rebuild with data_splitting/autovi_official_partitions.py.")

    pooled_train = np.concatenate(client_train)
    print(f"{name}: {len(files)} clients, pooled train {len(pooled_train)} normal images, "
          f"test {len(y_test)} images ({int((y_test == 1).sum())} anomalies), device {device}")

    per_seed: list[dict] = []
    for seed in args.seeds:
        set_seed(seed)
        run_dir = exp_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_metrics: dict[str, dict] = {}

        # --- pooled arm ----------------------------------------------------
        model = PaDiM(n_dims=args.n_dims, seed=seed, device=device,
                      inv_chunk=args.inv_chunk)
        model.fit(pooled_train, batch_size=args.batch_size)
        val_normals = np.concatenate([x[y == 0] for x, y in client_val])
        val_scores, _ = model.score(val_normals, batch_size=args.batch_size,
                                    sigma=args.sigma, return_maps=False)
        threshold = float(np.percentile(val_scores, NORMAL_QUANTILE))
        scores, maps = model.score(x_test, batch_size=args.batch_size, sigma=args.sigma,
                                   return_maps=not args.no_heatmaps)
        seed_metrics["pooled"] = metrics_from_scores(y_test, scores, threshold)
        print(f"  seed {seed} pooled      AUROC {seed_metrics['pooled']['auroc']:.3f}  "
              f"AP {seed_metrics['pooled']['average_precision']:.3f}")

        if maps is not None:
            n = save_heatmaps(run_dir, x_test, maps, y_test, scores,
                              per_class=args.heatmaps_per_class,
                              pick=args.heatmap_pick)
            print(f"    {n} heatmaps -> {run_dir / 'heatmaps'}")
        del maps

        # One score figure, from the first seed only: it illustrates the score
        # distribution, and three near-identical copies would say nothing more.
        if args.figures and seed == args.seeds[0]:
            fig_path = REPO_ROOT / "outputs" / "figures" / f"{name}_scores.png"
            if save_score_figure(fig_path, y_test, scores, threshold,
                                 f"PaDiM pooled, test set, seed {seed}"):
                print(f"    score figure -> {fig_path}")

        # --- per-client arms -----------------------------------------------
        # Two arms per client, on the SAME images, differing in exactly one variable:
        #   client_<i>            the LOCAL model (fitted on client i's normals only)
        #   central_on_client<i>  the POOLED model, scored on that same test slice
        # That one-variable difference is what makes "would this site be better off
        # alone?" answerable. per_client_mean vs pooled cannot answer it: it moves the
        # training set AND the evaluation set at once, so a gap there could just as
        # well mean one client's slice is easier than another's.
        #
        # Why the test slice and not each client's own validation set: under the
        # official AutoVI protocol EVERY anomaly goes to test, so the clients' val
        # splits are normals-only (43/29/29 images here, zero positives) and no AUROC
        # is computable on them at all. The owner-labelled test slices have both
        # classes and are much larger (547/324/305). Thresholds are still taken from
        # validation normals only, so no test label influences any threshold.
        client_scores = np.full(len(y_test), np.nan)
        central_keys = []
        for client_id, (x_tr, (x_va, y_va)) in enumerate(zip(client_train, client_val)):
            if owner is None:
                break
            mask = owner == client_id
            if not mask.any():
                continue

            m = PaDiM(n_dims=args.n_dims, seed=seed, device=device,
                      inv_chunk=args.inv_chunk)
            m.fit(x_tr, batch_size=args.batch_size)
            # Each arm is thresholded by ITS OWN model's validation normals: the two
            # models' score scales differ by construction, so one shared threshold
            # would make the thresholded columns incomparable. thr_loc is the local
            # model's; `threshold` (above) is the pooled model's.
            v_loc, _ = m.score(x_va[y_va == 0], batch_size=args.batch_size,
                               sigma=args.sigma, return_maps=False)
            thr_loc = float(np.percentile(v_loc, NORMAL_QUANTILE))

            s_i, _ = m.score(x_test[mask], batch_size=args.batch_size,
                             sigma=args.sigma, return_maps=False)
            client_scores[mask] = s_i
            key = f"client_{client_id}"
            seed_metrics[key] = {"samples": int(mask.sum()),
                                 **metrics_from_scores(y_test[mask], s_i, thr_loc)}

            # The pooled model's scores on these same images are already computed.
            ck = f"central_on_client{client_id}"
            seed_metrics[ck] = {"samples": int(mask.sum()),
                                **metrics_from_scores(y_test[mask], scores[mask], threshold)}
            central_keys.append(ck)

            delta = seed_metrics[key]["auroc"] - seed_metrics[ck]["auroc"]
            print(f"  seed {seed} {key} ({int(mask.sum())} imgs)  "
                  f"local AUROC {seed_metrics[key]['auroc']:.3f}  vs  "
                  f"central {seed_metrics[ck]['auroc']:.3f}  "
                  f"(local {'+' if delta >= 0 else ''}{delta:.3f})")

        # Exact key match, not startswith("client_"): the new arms contain "client"
        # too, and averaging them into per_client_mean would double-count.
        client_keys = [f"client_{i}" for i in range(len(files))
                       if f"client_{i}" in seed_metrics]
        if client_keys:
            seed_metrics["per_client_mean"] = {
                metric: float(np.nanmean([seed_metrics[k][metric] for k in client_keys]))
                for metric in METRICS
            }
            print(f"  seed {seed} per-client  AUROC "
                  f"{seed_metrics['per_client_mean']['auroc']:.3f} (mean over clients)")
        # The pooled model's mean over the same per-client slices, so it can be put
        # next to per_client_mean: same images on both sides, only training differs.
        if central_keys:
            seed_metrics["central_on_own_slice_mean"] = {
                metric: float(np.nanmean([seed_metrics[k][metric] for k in central_keys]))
                for metric in METRICS
            }
            d = (seed_metrics["per_client_mean"]["auroc"]
                 - seed_metrics["central_on_own_slice_mean"]["auroc"])
            print(f"  seed {seed} local-vs-central on identical slices: "
                  f"{seed_metrics['per_client_mean']['auroc']:.3f} vs "
                  f"{seed_metrics['central_on_own_slice_mean']['auroc']:.3f} "
                  f"(local {'+' if d >= 0 else ''}{d:.3f} AUROC)")

        run_config = {"partitions-dir": str(partitions), "seed": seed, "mode": "padim",
                      "n-dims": args.n_dims, "sigma": args.sigma,
                      "device": str(device), "batch-size": args.batch_size,
                      "cov-eps": 0.01, "backbone": "resnet18-imagenet1k-v1",
                      "threshold-rule": f"{NORMAL_QUANTILE}th pct of validation normals",
                      "protocol": "official (normal-only train)"}
        (run_dir / "metrics.json").write_text(json.dumps(
            {"run_config": run_config, "metrics": seed_metrics}, indent=2), encoding="utf-8")
        # Raw per-image scores, so Member 3 can build ROC curves without re-running.
        (run_dir / "scores.json").write_text(json.dumps({
            "label": y_test.tolist(),
            "owner": owner.tolist() if owner is not None else None,
            "pooled_score": scores.tolist(),
            "per_client_score": [None if np.isnan(v) else float(v) for v in client_scores],
            "pooled_threshold": threshold,
        }, indent=2), encoding="utf-8")
        per_seed.append(seed_metrics)

    # --- mean +/- std over seeds -------------------------------------------
    arms = sorted({k for m in per_seed for k in m})
    summary = {
        "experiment": name,
        "partitions": str(partitions),
        "seeds": args.seeds,
        "protocol": "official (AutoVI unsupervised: normal-only train, all defects in test)",
        "arms": {
            arm: {
                metric: mean_std([m[arm][metric] for m in per_seed
                                  if arm in m and metric in m[arm]])
                for metric in METRICS
            }
            for arm in arms
        },
    }
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'arm':<26}{'AUROC':>16}{'AP':>16}{'bal.acc':>16}{'TNR':>16}")
    for arm in arms:
        a = summary["arms"][arm]

        def cell(metric: str) -> str:
            st = a[metric]
            return f"{st['mean']:>9.3f} +- {st['std']:<4.3f}"

        print(f"{arm:<26}{cell('auroc')}{cell('average_precision')}"
              f"{cell('balanced_accuracy_at_threshold')}{cell('tnr')}")
    # The threshold is the 95th pct of validation normals, so a test TNR far from
    # 0.95 means that calibration did not transfer to the test normals.
    print("\n(TNR is the calibration check: the threshold asks for 0.95 on validation "
          "normals.)")
    print(f"\nWrote {exp_dir}")


if __name__ == "__main__":
    main()
