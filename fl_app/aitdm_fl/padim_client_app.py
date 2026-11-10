"""Flower ClientApp for federated PaDiM: embed local normals, send sums, done.

One round, no training. The client sends per-patch sums of embeddings and outer
products - sufficient statistics - so the server can rebuild the pooled model
EXACTLY. No image and no per-image quantity leaves the client.

See docs/federated_padim.md.
Run it with the padim-* config keys; see fl_app/pyproject.toml.
"""

import numpy as np
from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from aitdm_fl.client_app import get_data_path
from aitdm_fl.padim_paths import add_experiments_to_path
from aitdm_fl.task import get_device, load_npz_arrays

app = ClientApp()


@app.train()
def collect_statistics(msg: Message, context: Context) -> Message:
    """Round 1 (and only): local sufficient statistics of this client's normals."""
    config = msg.content["config"]
    # Trap 1: the subset is chosen by the SERVER and used verbatim. Drawing it
    # locally would mix channel i of one client with channel j of another and
    # produce a meaningless model with no error at all.
    idx = np.asarray(config["channel-idx"], dtype=np.int64)
    n_dims = int(config["n-dims"])
    batch_size = int(config["batch-size"])

    # Imported here, not at module level: experiments/ is located from the run
    # config because Flower executes a COPY of this app (see padim_paths.py).
    add_experiments_to_path(str(context.run_config.get("partitions-dir", "")))
    from padim import PaDiM

    x, y = load_npz_arrays(get_data_path(context), "train")

    # The protocol guarantee, enforced here too: under the official AutoVI
    # protocol train/ has no defects. Training on anomalies teaches PaDiM that
    # defects are normal and collapses the AUROC (see padim_experiment.py).
    n_anom = int((y == 1).sum())
    if n_anom:
        raise RuntimeError(
            f"{n_anom} anomalies in this client's training split. PaDiM models "
            "NORMAL appearance. You are probably pointing at the supervised "
            "partitions - use data/partitions/autovi_official.")
    if len(x) == 0:
        raise RuntimeError(f"{get_data_path(context)} has no training images.")

    model = PaDiM(n_dims=n_dims, device=get_device())
    stats = model.local_statistics(x, idx=idx, batch_size=batch_size)

    # float32 on the wire halves the payload (~82 MB -> ~41 MB at n_dims=100).
    # The float64 accumulation INSIDE local_statistics is what protects precision;
    # the server re-accumulates in float64 as well.
    wire = np.float32 if bool(config["float32-wire"]) else np.float64
    arrays = ArrayRecord({
        "S": Array(np.ascontiguousarray(stats["S"], dtype=wire)),
        "Q": Array(np.ascontiguousarray(stats["Q"], dtype=wire)),
    })
    metrics = MetricRecord({
        "num-examples": stats["n"],
        "grid-h": stats["grid"][0],
        "grid-w": stats["grid"][1],
    })
    return Message(content=RecordDict({"arrays": arrays, "metrics": metrics}),
                   reply_to=msg)
