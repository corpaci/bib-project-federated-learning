"""End-to-end check of the commands in the README, on small synthetic data.

Runs the real entry points as separate processes, exactly as a user would, with
the in-process federated backend (no Flower), so the same test passes on Windows,
Linux and macOS. No dataset download is needed.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_data import make_images  # noqa: E402

EXPERIMENTS = ROOT / "outputs" / "experiments"


def run(script: str, *args) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    return r


@pytest.fixture
def names():
    """Unique experiment names, removed again afterwards."""
    made = [f"smoke-{os.getpid()}-{tag}" for tag in ("a", "b")]
    yield made
    for name in made:
        shutil.rmtree(EXPERIMENTS / name, ignore_errors=True)


def official_like_partitions(out: Path, size: int = 64) -> None:
    """Three clients with normal-only train/val and a labelled global test set."""
    rng = np.random.default_rng(0)
    out.mkdir(parents=True)

    def images(n: int, client: int, defect: bool = False) -> np.ndarray:
        x = rng.normal(90 + 30 * client, 20, size=(n, size, size, 3))
        if defect:                                   # a bright square = the "defect"
            x[:, 20:40, 20:40] += 90
        return np.clip(x, 0, 255).astype(np.uint8)

    test_x, test_y, owner = [], [], []
    for c in range(3):
        np.savez_compressed(out / f"client_{c}.npz",
                            x_train=images(16, c), y_train=np.zeros(16, np.int64),
                            x_val=images(6, c), y_val=np.zeros(6, np.int64))
        test_x += [images(6, c), images(6, c, defect=True)]
        test_y += [np.zeros(6, np.int64), np.ones(6, np.int64)]
        owner += [np.full(12, c, np.int64)]
    np.savez_compressed(out / "test.npz", x_test=np.concatenate(test_x),
                        y_test=np.concatenate(test_y), owner_test=np.concatenate(owner))
    (out / "meta.json").write_text(json.dumps({"num_clients": 3, "num_classes": 2}),
                                   encoding="utf-8")


def test_padim_federated_matches_centralised(tmp_path, names):
    parts = tmp_path / "official"
    official_like_partitions(parts)
    central, federated = names

    run("experiments/padim_experiment.py", "--partitions", parts, "--n-dims", 10,
        "--seeds", 1, "--name", central, "--no-heatmaps")
    run("experiments/federated.py", "--model", "padim", "--partitions", parts,
        "--n-dims", 10, "--seeds", 1, "--name", federated)

    c = json.loads((EXPERIMENTS / central / "summary.json").read_text(encoding="utf-8"))
    f = json.loads((EXPERIMENTS / federated / "summary.json").read_text(encoding="utf-8"))
    pooled = c["arms"]["pooled"]["auroc"]["mean"]
    fed = f["arms"]["federated"]["auroc"]["mean"]
    assert pooled > 0.9, "the synthetic defect should be easy to detect"
    assert abs(pooled - fed) < 0.01, (pooled, fed)


def test_cnn_centralised_and_fedavg(tmp_path, names):
    make_images(tmp_path / "images.npz")
    parts = tmp_path / "labelled"
    run("data_splitting/split_data.py", "--input", tmp_path / "images.npz",
        "--num-clients", 3, "--strategy", "dirichlet", "--alpha", 1.0,
        "--min-samples", 5, "--out", parts)
    central, federated = names

    run("experiments/centralised.py", "--partitions", parts, "--epochs", 1,
        "--seeds", 1, "--name", central)
    run("experiments/federated.py", "--model", "cnn", "--partitions", parts,
        "--rounds", 2, "--local-epochs", 1, "--seeds", 1, "--name", federated)

    for name in (central, federated):
        s = json.loads((EXPERIMENTS / name / "summary.json").read_text(encoding="utf-8"))
        assert 0.0 <= s["test"]["balanced_accuracy"]["mean"] <= 1.0
        assert (EXPERIMENTS / name / "seed1" / "final_model.pt").exists()
