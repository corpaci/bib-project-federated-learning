"""Flower ServerApp for federated PaDiM: broadcast the channel subset, sum, invert.

NOT a Strategy. Aggregation here is EXACT and happens once, so there is nothing to
iterate and no FedAvg-shaped averaging to do. Averaging inv_cov would be wrong
(inv(mean) != mean(inv)): we sum the second moments and invert once, here.

Ignores num-server-rounds, local-epochs, learning-rate, strategy and the dp-* keys:
there are no gradients and no model updates to clip. See section 5 of
docs/federated_padim.md for why the existing differential-privacy wrapper
does not transfer.
"""

import json
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

from aitdm_fl.padim_paths import add_experiments_to_path
from aitdm_fl.task import get_device, set_seed

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    partitions_dir = str(cfg["partitions-dir"])
    if not partitions_dir:
        raise ValueError(
            "run config 'partitions-dir' is empty. Start the run with "
            "experiments/federated.py, which fills it in.")

    # Imported here, not at module level: experiments/ is located from
    # partitions-dir because Flower executes a COPY of this app (padim_paths.py).
    add_experiments_to_path(partitions_dir)
    from padim import PaDiM
    from padim_federated import aggregate_statistics, choose_channels

    set_seed(int(cfg["seed"]))
    n_dims = int(cfg["padim-n-dims"])

    # Trap 1: ONE channel subset for the whole federation, chosen here and sent to
    # everybody. Depends only on the seed, so it leaks nothing about any client's
    # data, and it matches what a centralised run with this seed would draw.
    idx = choose_channels(seed=int(cfg["seed"]), n_dims=n_dims)

    node_ids = list(grid.get_node_ids())
    min_nodes = int(cfg["min-nodes"])
    if len(node_ids) < min_nodes:
        raise RuntimeError(
            f"only {len(node_ids)} of {min_nodes} clients connected. PaDiM has ONE "
            "round, so a client that joins late contributes nothing at all.")

    config = ConfigRecord({
        "channel-idx": [int(i) for i in idx],
        "n-dims": n_dims,
        "batch-size": int(cfg["batch-size"]),
        "float32-wire": bool(cfg["padim-float32-wire"]),
    })
    messages = [
        Message(content=RecordDict({"arrays": ArrayRecord([]), "config": config}),
                message_type="train", dst_node_id=nid, group_id="1")
        for nid in node_ids
    ]

    stats, failures = [], []
    for reply in grid.send_and_receive(messages):
        if reply.has_error():
            failures.append(str(reply.error))
            continue
        arrays, m = reply.content["arrays"], reply.content["metrics"]
        stats.append({
            "n": int(m["num-examples"]),
            "S": arrays["S"].numpy(),
            "Q": arrays["Q"].numpy(),
            "grid": (int(m["grid-h"]), int(m["grid-w"])),
        })

    if failures:
        # One round means no recovery: with a single round there is no next round
        # to catch up in, so the model would be built on a subset of the federation
        # and nothing downstream would reveal it. Fail instead.
        raise RuntimeError(
            f"{len(failures)} of {len(node_ids)} client(s) failed: {failures}. "
            "Fix the clients and re-run.")

    agg = aggregate_statistics(stats)
    print(f"[padim] aggregated {len(stats)} clients, {agg['n']} normal images, "
          f"grid {agg['grid']}, n_dims {n_dims}")

    # Rebuild the pooled model from the sums alone - no images on the server.
    model = PaDiM(n_dims=n_dims, seed=int(cfg["seed"]), device=get_device())
    model.load_statistics(agg["mean"], agg["inv_cov"], idx, agg["grid"])

    from padim_experiment import evaluate_global_test
    results = evaluate_global_test(model, Path(partitions_dir),
                                   sigma=float(cfg["padim-sigma"]),
                                   batch_size=int(cfg["batch-size"]))
    results.update(n_clients=len(stats), n_train_images=agg["n"], n_dims=n_dims)
    print("[padim] global test: "
          + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in results.items()))

    save_dir = str(cfg["save-dir"])
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        # An .npz, not a .pt: there is no state_dict, only mean / inv_cov / idx.
        np.savez_compressed(out / "padim_federated_model.npz",
                            mean=agg["mean"], inv_cov=agg["inv_cov"],
                            idx=idx, grid=np.asarray(agg["grid"]))
        (out / "metrics.json").write_text(json.dumps(
            {"run_config": dict(cfg), "test": results}, indent=2), encoding="utf-8")
        print(f"\nSaved aggregated model and metrics to {out}")
