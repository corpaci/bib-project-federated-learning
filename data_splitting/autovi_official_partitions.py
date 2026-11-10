"""Build federated partitions that KEEP AutoVI's official unsupervised protocol.

Why this script exists
----------------------
`autovi.py --mode official` already produces the correct per-category split: train and
val hold NORMAL images only, every defect is in test. But its `--merged autovi_all.npz`
flattens those splits back into one pool (images / labels / category), and feeding that
pool to `split_data.py` makes it re-split stratified -- which puts anomalies straight
back into training and silently destroys the protocol.

So this script skips `split_data.py` and assembles the partitions directly from the
per-category npz files, one client per production part, preserving each category's
official train / val / test assignment.

    python data_splitting/autovi.py --zips data/raw/autovi/*.zip \\
        --out data/raw/autovi_official_npz --mode official --size 128
    python data_splitting/autovi_official_partitions.py \\
        --npz-dir data/raw/autovi_official_npz --out data/partitions/autovi_official

Result: client_<i>.npz with normal-only train/val, and a global test.npz holding every
defect of all three categories (568 of them) -- the images the unsupervised model is
scored against instead of trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ["normal", "anomaly"]
# One client per production part. Order fixes the client ids, so keep it stable.
CATEGORIES = ["engine_wiring", "pipe_clip", "pipe_staple"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated partitions preserving AutoVI's official (unsupervised) split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--npz-dir", type=Path, required=True,
                        help="output folder of autovi.py --mode official")
    parser.add_argument("--out", type=Path, required=True, help="partitions folder to write")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES,
                        help="one client per category, in this order")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    test_x, test_y, test_owner, clients = [], [], [], []

    for client_id, category in enumerate(args.categories):
        path = args.npz_dir / f"{category}.npz"
        if not path.exists():
            raise SystemExit(f"{path} not found. Run autovi.py --mode official first.")
        d = np.load(path)

        x_tr, y_tr = d["x_train"], d["y_train"].reshape(-1)
        x_va, y_va = d["x_val"], d["y_val"].reshape(-1)

        # The protocol guarantee, enforced rather than assumed.
        for split, y in (("train", y_tr), ("val", y_va)):
            n_anom = int((y == 1).sum())
            if n_anom:
                raise SystemExit(
                    f"{category}: {n_anom} anomalies in the {split} split. The npz was not "
                    "built with --mode official, or it came from a merged/re-split file.")

        np.savez_compressed(args.out / f"client_{client_id}.npz",
                            x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va)

        # Every defect of every category goes to the global test set.
        test_x.append(d["x_test"])
        test_y.append(d["y_test"].reshape(-1))
        test_owner.append(np.full(len(d["y_test"].reshape(-1)), client_id, dtype=np.int64))

        clients.append({
            "client": client_id,
            "category": category,
            "train": int(len(y_tr)),
            "val": int(len(y_va)),
            "train_class_counts": np.bincount(y_tr, minlength=2).tolist(),
            "test_class_counts": np.bincount(d["y_test"].reshape(-1), minlength=2).tolist(),
        })
        print(f"client {client_id} ({category}): train {len(y_tr)} normal-only, "
              f"val {len(y_va)} normal-only, test contributes "
              f"{int((d['y_test'].reshape(-1) == 1).sum())} defects")

    x_test = np.concatenate(test_x)
    y_test = np.concatenate(test_y)
    owner = np.concatenate(test_owner)
    # owner_test lets the per-client arm score each model on its own category.
    np.savez_compressed(args.out / "test.npz",
                        x_test=x_test, y_test=y_test, owner_test=owner)

    meta = {
        "source": str(args.npz_dir),
        "protocol": "official (AutoVI unsupervised: normal-only train, all defects in test)",
        "data_type": "images",
        "strategy": "group (one client per AutoVI production part)",
        "num_clients": len(args.categories),
        "num_classes": 2,
        "class_names": CLASS_NAMES,
        "sample_shape": list(x_test.shape[1:]),
        "test_samples": int(len(y_test)),
        "test_class_counts": np.bincount(y_test, minlength=2).tolist(),
        "clients": clients,
        "note": ("Built by autovi_official_partitions.py, NOT split_data.py: the merged npz "
                 "loses the official split and a stratified re-split would reintroduce "
                 "anomalies into training."),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    counts = np.bincount(y_test, minlength=2)
    print(f"\nGlobal test set: {len(y_test)} images, {counts[0]} normal, {counts[1]} anomaly")
    print(f"Saved {len(args.categories)} clients + test.npz to {args.out}")


if __name__ == "__main__":
    main()
