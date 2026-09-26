# Neural Architecture Comparison on Fashion-MNIST

A comparative study of four neural architectures — a feedforward ANN, a
convolutional CNN, a U-Net encoder–decoder, and a Vision Transformer —
trained and evaluated on the Fashion-MNIST image classification benchmark
(28×28 grayscale, 10 garment classes).

MSc coursework (De Montfort University), built as an applied-engineering
comparison: identical training protocol across models, with accuracy/loss
curves, confusion matrices, and prediction visualizations for every
architecture.

## Highlights

- **Head-to-head comparison** of four architectures on Fashion-MNIST
  (50,000 train / 10,000 validation / 10,000 test, seeded split).
- **CNN wins**: test accuracy **0.9072** and macro-F1 **0.9072** —
  convolution's translation-invariance inductive bias suits small grayscale
  images best.
- **Vision Transformer** (7×7 patches, 4 heads, 4 layers) reaches test
  accuracy 0.8686 — competitive with the ANN despite having no convolutional
  inductive bias, but converges slower.
- **U-Net** included as a structural experiment: 4-level encoder–decoder with
  skip connections, used for feature/reconstruction visualization rather than
  classification.
- Full per-class reports: **Shirt** is the hardest class across all models
  (F1 0.64–0.75); **Trouser** the easiest (F1 ≈ 0.98).

## Results

| Model | Test acc | Macro-F1 | Best val acc | Train time |
| --- | --- | --- | --- | --- |
| ANN (784→256→128→10) | 0.8713 | 0.8701 | 0.8881 | 2.26 min |
| **CNN (2 conv blocks)** | **0.9072** | **0.9072** | **0.9180** | 2.44 min |
| ViT (4 heads × 4 layers) | 0.8686 | 0.8669 | 0.8737 | 3.23 min |

(U-Net has no classification numbers — it was evaluated on feature
visualization, see "Architectures" below.)

Key insight: the CNN outperforms the MLP and ViT by ~3.5–4pp on test
accuracy despite being the smallest, most purpose-built architecture — on a
60k-image grayscale benchmark, spatial inductive bias beats raw capacity.
The ViT needs more data and longer training to realize its advantage on
small datasets.

## Architectures

All classification models trained for 12 epochs, Adam (lr=1e-3,
weight_decay=1e-5), CrossEntropyLoss, inputs normalized to [−1, 1].

- **ANN** — `Flatten → Linear(784→256) → ReLU → Dropout(0.3) →
  Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→10)`.
- **CNN** — `Conv2d(1→32, 3×3) → ReLU → MaxPool → Conv2d(32→64, 3×3) →
  ReLU → MaxPool → Linear(1600→256) → ReLU → Dropout(0.4) →
  Linear(256→10)`.
- **ViT** — convolutional patch embedding (patch size 7, embed dim 64),
  learned positional embeddings + CLS token, 4-layer Transformer encoder
  (4 attention heads), LayerNorm + linear classification head.
- **U-Net** — 4-level downsampling path (64→128→256→512 channels), 1024-dim
  bottleneck, transpose-convolution upsampling with skip connections and a
  final sigmoid layer. Visualized as "Original vs. U-Net Output" feature
  maps to inspect what the encoder–decoder path preserves.

## Repository contents

```
├── P2952028-SHEHRYAR SHEHRYAR.ipynb   # full experiment notebook (Colab/T4)
├── P2952028-A4_Summary (SHEHRYAR SHEHRYAR).pdf  # written A4 summary
├── Output SS/                         # output screenshots
└── README.md
```

## Getting started

The notebook is the complete experiment — open it in Google Colab (GPU
runtime recommended) and run top to bottom:

1. Environment & Fashion-MNIST download (50k/10k/10k seeded split)
2. Training loop utilities (`train_one_epoch`, `evaluate`, `fit`)
3. ANN training + accuracy/loss curves
4. CNN training + accuracy/loss curves
5. U-Net definition + feature-map visualization
6. ViT training + accuracy/loss curves
7. Test-set evaluation and per-class classification reports
8. Confusion matrices and prediction visualizations

PyTorch 2.8+ is required. The original run used a Colab T4 GPU; the models
are small enough to train on CPU in roughly 2–5× the listed times.

## Limitations

- Single dataset (Fashion-MNIST): conclusions about CNN vs. ViT inductive
  bias do not generalize to large-scale natural-image benchmarks.
- U-Net was not evaluated as a classifier, so its numbers are not comparable
  to the other three models.
- Hyperparameters were fixed across models (no per-architecture tuning), so
  results reflect default-protocol performance, not each architecture's best
  case.

## Author

**Shehryar** (P2952028) — MSc AI, De Montfort University.
GitHub: [@Shehryarcheema](https://github.com/Shehryarcheema)
