# Skin lesion classification on DermaMNIST – Group 6

Deep Learning challenge, M.Sc. Applied Data Science & AI, SRH University of Applied Sciences Fürth.
Team: Kaustubh Zanan, Manav Chauhan.

We classify 28 × 28 dermatoscopy images (DermaMNIST, 7 classes) with a small CNN. The main metric is
**balanced accuracy**, because the data is strongly imbalanced (67 % melanocytic nevi).

## Results

| Model | Val. balanced acc. (3 seeds) |
|---|---|
| Baseline CNN (starter) | 0.465 ± 0.021 |
| + Class weights | 0.535 ± 0.015 |
| + Augmentation (flips) | 0.450 ± 0.032 |
| + BatchNorm | 0.469 ± 0.037 |
| + Dropout 0.5 | 0.417 ± 0.017 |
| + LR scheduler (StepLR) | 0.476 ± 0.029 |
| CW + scheduler | 0.527 ± 0.018 |
| **CW + scheduler + BatchNorm (final)** | **0.566 ± 0.012** |

Submitted model (`model_group6.pt`, seed 0): **0.576** validation and **0.603** test balanced accuracy.

## Files

| File | What it is |
|---|---|
| `dermamnist_experiments.ipynb` | All experiments: data, baseline, 5 single changes, combinations, export of the final model |
| `model_group6.pt` | Final model (TorchScript): input normalised `(N, 3, 28, 28)`, output logits `(N, 7)` |
| `app.py` | Gradio app: upload an image → top-3 classes with probabilities |
| `requirements.txt` | Python packages |

## How to reproduce

```bash
pip install -r requirements.txt
```

1. Open `dermamnist_experiments.ipynb` and run all cells (**Run All**). The DermaMNIST data downloads
   automatically into `data/`. On a CPU the whole notebook takes about 5 minutes.
2. Every experiment is trained with seeds 0, 1 and 2, so the results are reproducible.
3. The last section chooses the final model on the validation set and exports it as `model.pt`.

## How to run the app

```bash
python app.py
```

A browser tab opens at http://127.0.0.1:7860. The app needs `model_group6.pt` in the same folder.

## Rules followed

Training only on the train split; all choices on the validation split; the test split was used once,
at the end, only to report the final model.

*Student project – not a medical device.*
