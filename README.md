# Industrial Anomaly Detection on AutoVI — federated learning project

Project for the course *AI for Trustworthy Decision Making*: detecting defective car
parts in assembly-line photos (Renault's AutoVI dataset), with a centralised baseline
and a federated version of the same model.

- **Stage 1 report:** [docs/stage1_report.md](docs/stage1_report.md)
- **Stage 1 slides (LaTeX):** [presentation/stage1_presentation.tex](presentation/stage1_presentation.tex)
- **Results behind every table:** [results/stage_01/](results/stage_01)

## Main results (Stage 1)

Mean ± standard deviation over seeds 1, 2, 3; three clients, one per production part.

| Model | Training | Metric | Value |
|---|---|---|---|
| PaDiM (unsupervised), official AutoVI split | centralised | test AUROC | 0.624 ± 0.017 |
| | federated, 1 round (Flower or in-process) | test AUROC | 0.624 ± 0.017 |
| CNN classifier (supervised), labelled split | centralised, 10 epochs | test balanced accuracy | 0.887 ± 0.008 |
| | FedAvg, 10 rounds (Flower) | test balanced accuracy | 0.849 ± 0.003 |

The federated PaDiM model is identical to the centralised one. The two models use
different splits and metrics and must not be compared with each other; the report
explains why.

## What is in this repository

| Path | Content | Report section |
|---|---|---|
| `data_splitting/autovi.py` | AutoVI zips → arrays (duplicates removed, RGB, resized) | 3 |
| `data_splitting/autovi_official_partitions.py` | 3 clients keeping AutoVI's official split | 3 |
| `data_splitting/split_data.py`, `partitioning.py` | Labelled split and the non-IID stress-test splits | 3 |
| `notebooks/stage_01/data_lead.ipynb` | Dataset analysis and how different the clients are | 3 |
| `experiments/padim.py` | The PaDiM model | 4 |
| `notebooks/stage_01/baseline_model_with_markdowns.ipynb` | PaDiM step by step, on the labelled split | 4, 6.2 |
| `experiments/padim_experiment.py` | PaDiM: centralised and per-client runs | 6.1, 6.2 |
| `experiments/padim_federated.py` | Aggregation of the clients' PaDiM statistics | 5 |
| `experiments/federated.py` | Federated runs (PaDiM and FedAvg), two backends | 5, 6.1, 6.3 |
| `experiments/centralised.py` | CNN classifier, centralised | 6.3 |
| `experiments/evaluation.py`, `summarize.py` | Shared evaluation and mean ± std tables for the CNN | 6.3 |
| `fl_app/` | The Flower app (ServerApp and ClientApp) | 5 |
| `docs/federated_padim.md` | Why federated PaDiM equals pooled PaDiM | 5 |
| `tests/` | Unit tests and an end-to-end test on synthetic data | — |

No AutoVI image is stored here (licence CC BY-NC-SA 4.0); `data/` and `outputs/` are
git-ignored.

## Setup

Python 3.11 or 3.12. All commands are the same on Windows, Linux and macOS, except
the line that activates the environment. Run them from the repository folder.

```text
python -m venv .venv

.venv\Scripts\activate           (Windows, PowerShell or cmd)
source .venv/bin/activate        (Linux, macOS)

pip install -r requirements.txt
python -m pytest -q tests
```

The tests need no dataset. The first run downloads the ResNet18 weights (45 MB).

**Operating system notes**

- Everything in this README runs natively on Windows with the default
  `--backend inprocess`.
- `--backend flower` needs `pip install -r requirements-flower.txt`. Flower officially
  supports Linux and macOS; on Windows use it inside WSL2.
- The repository has been run end to end on macOS. Windows and Linux are covered by
  the GitHub Actions workflow in `.github/workflows/ci.yml`, which runs the tests on
  both systems on every push.

## Reproduce Stage 1

### 1. Get the data

Download `engine_wiring.zip`, `pipe_clip.zip` and `pipe_staple.zip` from
<https://doi.org/10.5281/zenodo.10459003> into `data/raw/autovi/`.

### 2. PaDiM: centralised and federated (official split)

```text
python data_splitting/autovi.py --zips data/raw/autovi/pipe_staple.zip data/raw/autovi/pipe_clip.zip data/raw/autovi/engine_wiring.zip --out data/raw/autovi_official_npz --mode official --size 128
python data_splitting/autovi_official_partitions.py --npz-dir data/raw/autovi_official_npz --out data/partitions/autovi_official

python experiments/padim_experiment.py --partitions data/partitions/autovi_official --name padim-centralised --figures
python experiments/federated.py --model padim --partitions data/partitions/autovi_official --name padim-federated
```

About 2 minutes each on a laptop CPU. Each run prints its table and writes
`outputs/experiments/<name>/summary.json`.

### 3. CNN: centralised and FedAvg (labelled split)

```text
python data_splitting/autovi.py --zips data/raw/autovi/pipe_staple.zip data/raw/autovi/pipe_clip.zip data/raw/autovi/engine_wiring.zip --out data/raw/autovi_npz --size 128 --merged data/raw/autovi_npz/autovi_all.npz
python data_splitting/split_data.py --input data/raw/autovi_npz/autovi_all.npz --num-clients 3 --class-names "normal,anomaly" --seed 42 --strategy group --group-col category --out data/partitions/autovi

python experiments/centralised.py --partitions data/partitions/autovi --epochs 10 --name cnn-centralised
python experiments/federated.py --model cnn --partitions data/partitions/autovi --rounds 10 --local-epochs 1 --name cnn-fedavg
python experiments/summarize.py outputs/experiments/cnn-centralised outputs/experiments/cnn-fedavg --with-test
```

Keep the three zips in the order shown: the labelled split draws its test set by
image position, so another order gives a different (equally valid) split and slightly
different numbers.

### 4. The same federated runs through Flower (Linux, macOS, WSL2)

```text
pip install -r requirements-flower.txt
python experiments/federated.py --model padim --backend flower --partitions data/partitions/autovi_official --name padim-federated-flower
python experiments/federated.py --model cnn --backend flower --partitions data/partitions/autovi --name cnn-fedavg-flower
```

### 5. Notebooks

```text
jupyter notebook notebooks/stage_01
```

- `baseline_model_with_markdowns.ipynb` needs step 3's `data/partitions/autovi`.
  Its saved outputs were produced with exactly those commands.
- `data_lead.ipynb` analyses the images at their original 400 × 400 size. Its saved
  outputs come from the university cluster. To re-run it, rebuild the labelled data
  at that size first (this replaces the 128-pixel version from step 3; about 1 GB):

```text
python data_splitting/autovi.py --zips data/raw/autovi/pipe_staple.zip data/raw/autovi/pipe_clip.zip data/raw/autovi/engine_wiring.zip --out data/raw/autovi_npz --size 400 --keep-extracted --merged data/raw/autovi_npz/autovi_all.npz
python data_splitting/split_data.py --input data/raw/autovi_npz/autovi_all.npz --num-clients 3 --class-names "normal,anomaly" --seed 42 --strategy group --group-col category --out data/partitions/autovi
python experiments/centralised.py --partitions data/partitions/autovi --seeds 1 --name member1-centralised
```

### 6. Slides

```text
cd presentation
pdflatex stage1_presentation.tex
pdflatex stage1_presentation.tex
```

Or upload the `presentation/` folder to Overleaf. Logos are optional: files named
`upb.png`, `acs.png` in `presentation/logos/` appear on the first and last slide.

## Dataset and licence

AutoVI © Renault Group, CC BY-NC-SA 4.0 (non-commercial).
Carvalho, P., Lafou, M., Durupt, A., Leblanc, A., Grandvalet, Y. (2024). *Detecting
visual anomalies in an industrial environment: Unsupervised methods put to the test on
the AutoVI dataset.* Computers in Industry 163, 104151.

The federated learning code builds on the course starter kit
(AITDM Flower project tutorial).
