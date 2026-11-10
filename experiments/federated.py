"""Federated runs over several seeds, on Windows, Linux and macOS.

    # PaDiM: one round, exact aggregation of the clients' statistics
    python experiments/federated.py --model padim --partitions data/partitions/autovi_official

    # CNN classifier: FedAvg, 10 rounds of 1 local epoch
    python experiments/federated.py --model cnn --partitions data/partitions/autovi

Two backends run the same client and server computations:

  --backend inprocess   (default) the clients are simulated one after the other in
                        this Python process. Needs only PyTorch, so it works on
                        every operating system, including native Windows.
  --backend flower      the Flower app in fl_app/ (ServerApp + one ClientApp per
                        client, started with `flwr run`). Needs
                        requirements-flower.txt. Flower supports Linux and macOS;
                        on Windows use it inside WSL2.

Results go to outputs/experiments/<name>/seed<k>/ and a mean +- std over the seeds
to outputs/experiments/<name>/summary.json, in the same format as the centralised
runners (padim_experiment.py, centralised.py), so the tables can be put side by side.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "fl_app"))

from aitdm_fl.task import (  # noqa: E402
    balanced_class_weights,
    build_model,
    derive_seed,
    get_device,
    load_client_data,
    load_npz_arrays,
    model_from_test_set,
    num_classes_of_dataset,
    set_seed,
    train,
)


def client_files(partitions: Path) -> list[Path]:
    files = sorted(partitions.glob("client_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not files or not (partitions / "test.npz").exists():
        raise SystemExit(f"no client_*.npz / test.npz in {partitions}. Build the "
                         "partitions first, see README.md.")
    return files


# ---------------------------------------------------------------------------
# PaDiM: every client sends (n, S, Q); the server adds them and inverts once
# ---------------------------------------------------------------------------
def padim_inprocess(partitions: Path, seed: int, run_dir: Path, args) -> dict:
    from padim import PaDiM
    from padim_experiment import check_normal_only, evaluate_global_test
    from padim_federated import aggregate_statistics, choose_channels

    set_seed(seed)
    device = get_device()
    # Server: ONE channel subset for the whole federation, a function of the seed only.
    idx = choose_channels(seed=seed, n_dims=args.n_dims)

    # Clients: statistics of their own normal images. Nothing else leaves a client.
    stats = []
    for path in client_files(partitions):
        x, y = load_npz_arrays(path, "train")
        check_normal_only(y, partitions, f"training split of {path.name}")
        client = PaDiM(n_dims=args.n_dims, device=device)
        stats.append(client.local_statistics(x, idx=idx, batch_size=args.batch_size))
        del client

    # Server: add the sums, invert once, evaluate on the global test set.
    agg = aggregate_statistics(stats)
    model = PaDiM(n_dims=args.n_dims, seed=seed, device=device)
    model.load_statistics(agg["mean"], agg["inv_cov"], idx, agg["grid"])
    results = evaluate_global_test(model, partitions, sigma=args.sigma,
                                   batch_size=args.batch_size)
    results.update(n_clients=len(stats), n_train_images=agg["n"], n_dims=args.n_dims)

    np.savez_compressed(run_dir / "padim_federated_model.npz", mean=agg["mean"],
                        inv_cov=agg["inv_cov"], idx=idx, grid=np.asarray(agg["grid"]))
    run_config = {"partitions-dir": str(partitions), "seed": seed, "mode": "padim-federated",
                  "backend": "inprocess", "n-dims": args.n_dims, "sigma": args.sigma,
                  "batch-size": args.batch_size}
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_config": run_config, "test": results}, indent=2), encoding="utf-8")
    return results


def summarize_padim(exp_dir: Path, name: str, partitions: Path, per_seed: dict[int, dict],
                    backend: str) -> dict:
    from padim_experiment import METRICS, mean_std

    summary = {
        "experiment": name,
        "partitions": str(partitions),
        "seeds": sorted(per_seed),
        "backend": backend,
        "arms": {"federated": {m: mean_std([per_seed[s][m] for s in sorted(per_seed)])
                               for m in METRICS}},
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    a = summary["arms"]["federated"]
    print(f"\n{name} ({len(per_seed)} seeds, mean +- std)")
    print(f"  test  AUROC {a['auroc']['mean']:.3f} +- {a['auroc']['std']:.3f}, "
          f"AP {a['average_precision']['mean']:.3f} +- {a['average_precision']['std']:.3f}, "
          f"bal. acc. {a['balanced_accuracy_at_threshold']['mean']:.3f} +- "
          f"{a['balanced_accuracy_at_threshold']['std']:.3f}")
    return summary


# ---------------------------------------------------------------------------
# CNN: FedAvg (McMahan et al., 2017)
# ---------------------------------------------------------------------------
def fedavg(states: list[dict], sizes: list[int]) -> dict:
    """Sample-weighted average of the clients' model weights."""
    total = float(sum(sizes))
    out = {}
    for key, first in states[0].items():
        if torch.is_floating_point(first):
            out[key] = sum(s[key].double() * (n / total)
                           for s, n in zip(states, sizes)).to(first.dtype)
        else:                       # integer buffers (e.g. batch counters): not averaged
            out[key] = first.clone()
    return out


def cnn_inprocess(partitions: Path, seed: int, run_dir: Path, args) -> None:
    files = client_files(partitions)
    num_classes = num_classes_of_dataset(str(partitions))
    device = get_device()

    set_seed(seed)                                  # same seed -> same initial model
    global_state = {k: v.cpu().clone()
                    for k, v in model_from_test_set(str(partitions)).state_dict().items()}

    history = {}
    for rnd in range(1, args.rounds + 1):
        states, sizes, losses = [], [], []
        for path in files:
            # Same per-client, per-round seed rule as the Flower ClientApp.
            client_seed = derive_seed(seed, path.name, rnd)
            set_seed(client_seed)
            train_loader, _ = load_client_data(path, args.batch_size, seed=client_seed)
            model = build_model(sample_shape=tuple(train_loader.dataset.x.shape[1:]),
                                num_classes=num_classes)
            model.load_state_dict(global_state)
            weights = None
            if args.loss_weighting == "balanced":
                weights = balanced_class_weights(train_loader.dataset.y.numpy(), num_classes)
            loss = train(model, train_loader, epochs=args.local_epochs, lr=args.lr,
                         device=device, class_weights=weights)
            states.append({k: v.cpu() for k, v in model.state_dict().items()})
            sizes.append(len(train_loader.dataset))
            losses.append(loss)
        global_state = fedavg(states, sizes)
        mean_loss = float(np.average(losses, weights=sizes))
        history[rnd] = {"train_loss": mean_loss}
        print(f"  seed {seed} round {rnd:>2}/{args.rounds}: train loss {mean_loss:.4f}", flush=True)

    torch.save(global_state, run_dir / "final_model.pt")
    run_config = {"partitions-dir": str(partitions), "seed": seed, "mode": "fedavg",
                  "backend": "inprocess", "num-server-rounds": args.rounds,
                  "local-epochs": args.local_epochs, "learning-rate": args.lr,
                  "batch-size": args.batch_size, "loss-weighting": args.loss_weighting,
                  "strategy": "fedavg"}
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_config": run_config, "train_metrics": history}, indent=2),
        encoding="utf-8")


# ---------------------------------------------------------------------------
# Flower backend: the same computations, run as a Flower app
# ---------------------------------------------------------------------------
def toml_string(text: str) -> str:
    """A double-quoted TOML string. Handles Windows backslashes and apostrophes."""
    return json.dumps(str(text), ensure_ascii=False)


def run_flower(partitions: Path, seed: int, run_dir: Path, args) -> None:
    exe = shutil.which("flwr") or str(Path(sys.executable).parent / "flwr")
    if not shutil.which(exe):
        raise SystemExit("the 'flwr' command was not found. Install the Flower backend "
                         "with: pip install -r requirements-flower.txt  "
                         "(or use --backend inprocess)")
    n = len(client_files(partitions))
    config = [f"partitions-dir={toml_string(partitions)}", f"save-dir={toml_string(run_dir)}",
              f"min-nodes={n}", f"seed={seed}", f"batch-size={args.batch_size}"]
    if args.model == "padim":
        config += ["padim-mode=true", "num-server-rounds=1", "fraction-evaluate=0.0",
                   f"padim-n-dims={args.n_dims}", f"padim-sigma={float(args.sigma)}"]
    else:
        config += [f"num-server-rounds={args.rounds}", f"local-epochs={args.local_epochs}",
                   f"learning-rate={args.lr}", "strategy='fedavg'",
                   f"loss-weighting='{args.loss_weighting}'"]
    env = {**os.environ,
           "AITDM_REPO_ROOT": str(REPO_ROOT),      # lets the app find experiments/padim.py
           "FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION": "1",
           "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0"}
    log = run_dir.parent / f"{run_dir.name}.log"
    with open(log, "w", encoding="utf-8") as out:
        # A list of arguments, no shell: nothing to quote or escape per platform.
        subprocess.run([exe, "run", ".", "--stream",
                        "--federation-config", f"num-supernodes={n}",
                        "--run-config", " ".join(config)],
                       cwd=REPO_ROOT / "fl_app", env=env, stdout=out,
                       stderr=subprocess.STDOUT, check=False)
    # `flwr run` does not fail when the ServerApp crashes: check for the results.
    if not (run_dir / "metrics.json").exists():
        raise SystemExit(f"The Flower run failed, see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated runs (PaDiM statistics aggregation, or FedAvg for the CNN).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=["padim", "cnn"], required=True)
    parser.add_argument("--partitions", type=Path, required=True,
                        help="padim: official-protocol partitions (normal-only train). "
                             "cnn: labelled partitions written by split_data.py")
    parser.add_argument("--backend", choices=["inprocess", "flower"], default="inprocess")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--name", help="experiment name (default: <model>-federated-<partitions>)")
    parser.add_argument("--batch-size", type=int, default=32)
    padim = parser.add_argument_group("padim")
    padim.add_argument("--n-dims", type=int, default=100, help="channels kept out of 448")
    padim.add_argument("--sigma", type=float, default=4.0, help="blur of the anomaly map")
    cnn = parser.add_argument_group("cnn")
    cnn.add_argument("--rounds", type=int, default=10)
    cnn.add_argument("--local-epochs", type=int, default=1)
    cnn.add_argument("--lr", type=float, default=0.001)
    cnn.add_argument("--loss-weighting", choices=["none", "balanced"], default="none")
    args = parser.parse_args()

    partitions = args.partitions.resolve()
    client_files(partitions)                        # fail early, with a clear message
    name = args.name or f"{args.model}-federated-{partitions.name}"
    exp_dir = REPO_ROOT / "outputs" / "experiments" / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    padim_results: dict[int, dict] = {}
    for seed in args.seeds:
        run_dir = exp_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f">>> {name} seed {seed} ({args.backend})", flush=True)
        if args.backend == "flower":
            run_flower(partitions, seed, run_dir, args)
        elif args.model == "padim":
            padim_inprocess(partitions, seed, run_dir, args)
        else:
            cnn_inprocess(partitions, seed, run_dir, args)

        if args.model == "padim":
            test = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["test"]
            padim_results[seed] = test
            print(f"  seed {seed}: test AUROC {test['auroc']:.4f}")
        else:
            from evaluation import write_results
            result = write_results(run_dir, partitions)
            print(f"  seed {seed}: val balanced acc {result['val']['balanced_accuracy']:.3f}, "
                  f"test balanced acc {result['test']['balanced_accuracy']:.3f}")

    if args.model == "padim":
        summarize_padim(exp_dir, name, partitions, padim_results, args.backend)
    else:
        from summarize import summarize_experiment
        summarize_experiment(exp_dir)
    print(f"\nWrote {exp_dir}")


if __name__ == "__main__":
    main()
