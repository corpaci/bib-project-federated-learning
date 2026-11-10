# Federated PaDiM: one round, exact aggregation

This page explains how the project trains PaDiM across several clients and why the
result is the same model a single site would get from all the images together.
It is the reference for `experiments/padim_federated.py`, `experiments/federated.py`,
`fl_app/aitdm_fl/padim_server_app.py` and `fl_app/aitdm_fl/padim_client_app.py`.

## 1. What PaDiM stores

PaDiM (Defard et al., 2021) has no trainable weights. A frozen, ImageNet-pretrained
ResNet18 turns each image into a grid of patch embeddings. For every grid position
`p` the model keeps two things, estimated from the **normal** training images only:

- the mean embedding `mu_p` (a vector of `d` numbers),
- the covariance `Sigma_p` (a `d x d` matrix).

A test patch is scored by its Mahalanobis distance to `(mu_p, Sigma_p)`; the image
score is the maximum over the grid.

## 2. Why federation is exact

Mean and covariance are computed from two sums over the `n` training images,
with `x_i` the embedding of image `i` at position `p`:

```text
S = sum_i x_i            (sum of embeddings)
Q = sum_i x_i x_i^T      (sum of outer products)

mu    = S / n
Sigma = (Q / n - mu mu^T) * n / (n - 1)
```

A sum over all images equals the sum of the per-client sums. So if each client `k`
sends `(n_k, S_k, Q_k)` and the server adds them, the server holds exactly the `n`,
`S` and `Q` of the pooled data, and therefore exactly the pooled model. No image and
no per-image value leaves a client.

This is a sample-weighted average of client statistics, the same weighting FedAvg
uses, applied to `S/n` and `Q/n` instead of to network weights. The difference from
FedAvg on a neural network is that one round is enough and nothing is approximated.

## 3. Three things that must be done correctly

1. **One channel subset for everybody.** PaDiM keeps a random subset of `d = 100`
   out of 448 embedding channels. The server draws it from the seed and sends it to
   all clients (`choose_channels`). If clients drew their own, the summed statistics
   would mix unrelated channels and the model would be meaningless.
2. **Sum first, invert once.** Clients never send an inverse covariance: the average
   of inverses is not the inverse of the average. The server inverts after summing.
3. **Same input size on every client.** The patch grid depends on the image size;
   the server refuses to aggregate clients whose grids differ.

`tests/test_padim_federated.py` checks all three, and checks that the federated and
pooled models agree. The aggregation is exact; with the real ResNet18 the two models
can still differ by about 1e-6 relative, because the float32 network returns slightly
different features when the same image is processed in a different batch.

## 4. Run it

PaDiM needs partitions whose training split contains normal images only
(the official AutoVI protocol):

```text
python data_splitting/autovi.py --zips data/raw/autovi/pipe_staple.zip data/raw/autovi/pipe_clip.zip data/raw/autovi/engine_wiring.zip --out data/raw/autovi_official_npz --mode official --size 128
python data_splitting/autovi_official_partitions.py --npz-dir data/raw/autovi_official_npz --out data/partitions/autovi_official

python experiments/padim_experiment.py --partitions data/partitions/autovi_official --name padim-centralised --figures
python experiments/federated.py --model padim --partitions data/partitions/autovi_official --name padim-federated
```

The federated run writes `metrics.json` and the aggregated model to
`outputs/experiments/padim-federated/seed<k>/`. Its test AUROC matches the `pooled`
arm of `padim_experiment.py` for the same seed. Add `--backend flower` to run the
same computation as a Flower app (`fl_app/`: one ServerApp, one ClientApp per client).

## 5. What this setup does not give you

- **The gradient-based trust tools do not apply.** `num-server-rounds`,
  `local-epochs`, `learning-rate`, `strategy` and every `dp-*` and attack key of the
  Flower app are ignored in `padim-mode`. There are no model updates to clip, so the Flower
  differential-privacy wrapper cannot be reused; noise would have to be added to
  `S` and `Q` directly.
- **Statistics are not private by default.** `S_k / n_k` is the client's mean
  embedding. A client with very few images reveals more about each of them.
- **One round means no recovery.** A client that fails or joins late contributes
  nothing, so the server stops with an error instead of building a partial model.
- **A wrong client is not detected.** A client that sends statistics of defective
  images shifts the pooled model. This is a robustness question for Stage 2.

## References

- Defard, T., Setkov, A., Loesch, A., Audigier, R. (2021). *PaDiM: a Patch
  Distribution Modeling Framework for Anomaly Detection and Localization.*
  ICPR Workshops. arXiv:2011.08785.
- McMahan, H. B., et al. (2017). *Communication-Efficient Learning of Deep Networks
  from Decentralized Data.* AISTATS.
- Carvalho, P., Lafou, M., Durupt, A., Leblanc, A., Grandvalet, Y. (2024).
  *Detecting visual anomalies in an industrial environment: Unsupervised methods put
  to the test on the AutoVI dataset.* Computers in Industry 163, 104151.
