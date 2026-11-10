"""AutoVI (Automotive Visual Inspection Dataset) -> the npz format this repo uses.

AutoVI is Renault Group's real production-line anomaly dataset:
https://auto-vi.utc.fr/  |  https://doi.org/10.5281/zenodo.10459003
Carvalho, Lafou, Durupt, Leblanc & Grandvalet (2024), Computers in Industry.
Licence: CC BY-NC-SA 4.0 (non-commercial). NOT redistributable - keep it out of git.

On disk each object category is one zip with this layout (MVTec-AD style):

    <object>/train/good/0000.png              normal images ONLY
    <object>/test/good/0000.png               normal images
    <object>/test/<defect>/0000.png           defective images
    <object>/ground_truth/<defect>/0000/*.png segmentation masks

IMPORTANT - why this module re-splits the data
----------------------------------------------
AutoVI ships as an UNSUPERVISED benchmark: train/ holds no defects at all, because
the task it was designed for is "learn what normal looks like, then flag anything
else". A supervised classifier, which is what Stage 1 of this project builds, cannot
learn from that: it would never see a single positive example.

So `--mode supervised` (the default) pools every image of a category and re-splits it
stratified into train / val / test, so both classes appear in training. This is a
DELIBERATE DEVIATION from the official AutoVI protocol and must be stated in the
report: AUROC numbers produced this way are NOT comparable with published results.

`--mode official` keeps AutoVI's own split, for the faithful unsupervised setup.

Usage
-----
    # One npz per category (-> one federated client per category)
    python data_splitting/autovi.py --zips data/raw/autovi/*.zip --out data/raw/autovi_npz

    # Then partition by category: client = production line
    python data_splitting/split_data.py --input data/raw/autovi_npz/*.npz \\
        --strategy group --num-clients 3 --out data/partitions/autovi
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow")

NORMAL, ANOMALY = 0, 1
CLASS_NAMES = ["normal", "anomaly"]


def extract(zip_path: Path, dest: Path) -> Path:
    """Unzip <object>.zip into dest/ (skipped if already extracted)."""
    name = zip_path.stem
    target = dest / name
    if target.exists() and any(target.iterdir()):
        return target
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    if not target.exists():
        raise SystemExit(f"{zip_path} did not contain a folder named {name}/")
    return target


def load_image(path: Path, size: int) -> np.ndarray:
    """PNG (RGBA, 400x400) -> (size, size, 3) uint8."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def scan_category(root: Path) -> dict[str, list[Path]]:
    """Collect image paths by role: train_good, test_good, and each defect type."""
    found: dict[str, list[Path]] = {}

    train_good = sorted((root / "train" / "good").glob("*.png"))
    if train_good:
        found["train/good"] = train_good

    test_dir = root / "test"
    if test_dir.is_dir():
        for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            imgs = sorted(sub.glob("*.png"))
            if imgs:
                found[f"test/{sub.name}"] = imgs
    return found


def dedupe(groups: dict[str, list[Path]]) -> tuple[int, dict[str, int]]:
    """Drop byte-identical images, keeping the first copy seen.

    AutoVI contains exact duplicates (the same photograph filed under both
    train/good and test/good, for instance). Pooling and re-splitting would then put
    one copy in train and the other in test -- textbook leakage. Images are compared
    on their decoded RGB pixels, so two PNGs that differ only in metadata still count
    as the same image. Returns (number removed, {"folder_a <-> folder_b": pairs}).
    """
    seen: dict[str, str] = {}
    spans: Counter[str] = Counter()
    removed = 0
    for key in list(groups):            # dict order: train/good first, then test/*
        kept = []
        for path in groups[key]:
            with Image.open(path) as im:
                digest = hashlib.md5(np.asarray(im.convert("RGB")).tobytes()).hexdigest()
            if digest in seen:
                spans[f"{seen[digest]} <-> {key}"] += 1
                removed += 1
            else:
                seen[digest] = key
                kept.append(path)
        groups[key] = kept
    return removed, dict(spans)


def build_category(root: Path, size: int, mode: str, val_fraction: float,
                   test_fraction: float, seed: int) -> dict:
    """Return arrays + a report for one AutoVI object category."""
    groups = scan_category(root)
    if not groups:
        raise SystemExit(f"No PNGs found under {root}")
    raw_counts = {k: len(v) for k, v in groups.items()}
    n_dup, dup_spans = dedupe(groups)

    defect_names = sorted(k.split("/", 1)[1] for k in groups
                          if k.startswith("test/") and not k.endswith("/good"))

    report = {
        "category": root.name,
        "defect_types": defect_names,
        "counts_as_shipped": raw_counts,
        "exact_duplicates_removed": n_dup,
        "duplicate_pairs_by_folder": dup_spans,
        "counts": {k: len(v) for k, v in groups.items()},
        "mode": mode,
    }

    def to_array(paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, size, size, 3), dtype=np.uint8)
        return np.stack([load_image(p, size) for p in paths])

    rng = np.random.default_rng(seed)

    if mode == "official":
        # AutoVI's own split: train is normal-only (unsupervised).
        x_train = to_array(groups.get("train/good", []))
        y_train = np.full(len(x_train), NORMAL, dtype=np.int64)

        test_paths = list(groups.get("test/good", []))
        test_labels = [NORMAL] * len(test_paths)
        for name in defect_names:
            paths = groups[f"test/{name}"]
            test_paths += paths
            test_labels += [ANOMALY] * len(paths)
        x_test = to_array(test_paths)
        y_test = np.asarray(test_labels, dtype=np.int64)

        # Carve a validation set out of train (still normal-only)
        n_val = int(round(len(x_train) * val_fraction))
        perm = rng.permutation(len(x_train))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        x_val, y_val = x_train[val_idx], y_train[val_idx]
        x_train, y_train = x_train[tr_idx], y_train[tr_idx]

    else:  # supervised: pool everything, stratified re-split
        normal_paths = list(groups.get("train/good", [])) + list(groups.get("test/good", []))
        anomaly_paths: list[Path] = []
        for name in defect_names:
            anomaly_paths += groups[f"test/{name}"]

        splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}

        for paths, label in ((normal_paths, NORMAL), (anomaly_paths, ANOMALY)):
            idx = rng.permutation(len(paths))
            n_test = int(round(len(paths) * test_fraction))
            n_val = int(round(len(paths) * val_fraction))
            chunks = {
                "test": idx[:n_test],
                "val": idx[n_test:n_test + n_val],
                "train": idx[n_test + n_val:],
            }
            for split, ids in chunks.items():
                splits[split] += [paths[i] for i in ids]
                labels[split] += [label] * len(ids)

        x_train, y_train = to_array(splits["train"]), np.asarray(labels["train"], dtype=np.int64)
        x_val, y_val = to_array(splits["val"]), np.asarray(labels["val"], dtype=np.int64)
        x_test, y_test = to_array(splits["test"]), np.asarray(labels["test"], dtype=np.int64)

    report["split_sizes"] = {
        "train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test)),
    }
    report["anomaly_fraction"] = {
        "train": round(float(np.mean(y_train == ANOMALY)), 4) if len(y_train) else 0.0,
        "val": round(float(np.mean(y_val == ANOMALY)), 4) if len(y_val) else 0.0,
        "test": round(float(np.mean(y_test == ANOMALY)), 4) if len(y_test) else 0.0,
    }
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "report": report,
    }


def expand_wildcards(paths: list[Path]) -> list[Path]:
    """Expand * and ? in paths ourselves.

    Linux and macOS shells expand data/raw/autovi/*.zip before Python starts;
    Windows PowerShell and cmd pass the pattern through unchanged.
    """
    out: list[Path] = []
    for p in paths:
        if any(ch in str(p) for ch in "*?["):
            matches = sorted(Path(m) for m in glob.glob(str(p)))
            if not matches:
                raise SystemExit(f"no file matches {p}")
            out.extend(matches)
        else:
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AutoVI category zips into npz files this repo can split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--zips", type=Path, nargs="+", required=True,
                        help="AutoVI category zips, e.g. data/raw/autovi/*.zip")
    parser.add_argument("--out", type=Path, default=Path("data/raw/autovi_npz"),
                        help="output folder for the per-category npz files")
    parser.add_argument("--size", type=int, default=128,
                        help="images are resized to size x size (AutoVI ships 400x400)")
    parser.add_argument("--mode", choices=["supervised", "official"], default="supervised",
                        help="supervised: re-split so defects appear in training (needed for "
                             "a classifier). official: keep AutoVI's normal-only train split.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-extracted", action="store_true",
                        help="do not delete the unzipped images afterwards")
    parser.add_argument("--merged", type=Path,
                        help="also write ONE npz holding every category, with arrays "
                             "images / labels / category. split_data.py accepts only a "
                             "single npz, so this is what lets you make one federated "
                             "client per production part: --strategy group --group-col category")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_extracted"
    reports = []
    merged: list[dict] = []

    for zip_path in expand_wildcards(args.zips):
        if not zip_path.exists():
            raise SystemExit(f"{zip_path} not found")
        print(f">>> {zip_path.name}")
        root = extract(zip_path, work)
        data = build_category(root, args.size, args.mode,
                              args.val_fraction, args.test_fraction, args.seed)
        report = data.pop("report")
        out_npz = args.out / f"{root.name}.npz"
        np.savez_compressed(out_npz, **data)

        r = report
        print(f"    defects : {', '.join(r['defect_types']) or '(none)'}")
        print(f"    dedupe  : removed {r['exact_duplicates_removed']} byte-identical images "
              f"{r['duplicate_pairs_by_folder'] or ''}")
        print(f"    split   : train {r['split_sizes']['train']}, "
              f"val {r['split_sizes']['val']}, test {r['split_sizes']['test']}")
        print(f"    anomaly%: train {r['anomaly_fraction']['train']:.1%}, "
              f"val {r['anomaly_fraction']['val']:.1%}, "
              f"test {r['anomaly_fraction']['test']:.1%}")
        print(f"    saved   : {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB)")
        reports.append(r)
        if args.merged:
            merged.append({"category": root.name, **data})

    if args.merged:
        args.merged.parent.mkdir(parents=True, exist_ok=True)
        images = np.concatenate([np.concatenate([c["x_train"], c["x_val"], c["x_test"]])
                                 for c in merged])
        labels = np.concatenate([np.concatenate([c["y_train"], c["y_val"], c["y_test"]])
                                 for c in merged])
        category = np.concatenate([
            np.full(len(c["y_train"]) + len(c["y_val"]) + len(c["y_test"]), c["category"])
            for c in merged])
        np.savez_compressed(args.merged, images=images, labels=labels, category=category)
        print(f"\nMerged npz: {args.merged} ({args.merged.stat().st_size / 1e6:.1f} MB)")
        print(f"  images {images.shape}, {len(np.unique(category))} categories, "
              f"anomaly rate {labels.mean():.1%}")
        print("  -> one client per part:")
        print(f"     python data_splitting/split_data.py --input {args.merged} \\")
        print("         --strategy group --group-col category --num-clients "
              f"{len(np.unique(category))} --out data/partitions/autovi")

    summary = {
        "source": "AutoVI (Automotive Visual Inspection Dataset), Renault Group / UTC",
        "url": "https://auto-vi.utc.fr/",
        "doi": "10.5281/zenodo.10459003",
        "licence": "CC BY-NC-SA 4.0 (non-commercial, not redistributable)",
        "citation": ("Carvalho, P., Lafou, M., Durupt, A., Leblanc, A. & Grandvalet, Y. "
                     "(2024). The Automotive Visual Inspection Dataset (AutoVI). "
                     "Computers in Industry. https://doi.org/10.1016/j.compind.2024.104151"),
        "image_size": args.size,
        "mode": args.mode,
        "class_names": CLASS_NAMES,
        "seed": args.seed,
        "exact_duplicates_removed_total": sum(r["exact_duplicates_removed"] for r in reports),
        "categories": reports,
    }
    if args.mode == "supervised":
        summary["protocol_note"] = (
            "Data was RE-SPLIT so that defects appear in training. AutoVI's official "
            "protocol is unsupervised (train/ contains only normal images); results "
            "produced here are therefore NOT comparable with published AutoVI AUROC.")

    (args.out / "autovi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {args.out / 'autovi_summary.json'}")

    if not args.keep_extracted and work.exists():
        shutil.rmtree(work)
        print(f"Removed {work} (use --keep-extracted to keep it)")


if __name__ == "__main__":
    main()
