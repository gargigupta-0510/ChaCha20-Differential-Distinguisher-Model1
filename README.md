
# ChaCha20 Differential Distinguisher — 1D CNN Baseline

## Overview

This project explores the use of a one-dimensional
Convolutional Neural Network (1D CNN) to distinguish
differentially related inputs from random inputs for a
reduced-round ChaCha permutation.

## Model

The baseline model uses Conv1D layers, batch normalization,
pooling, dense layers and dropout.

## Project Files

- `chacha.py` — ChaCha permutation implementation
- `model.py` — CNN architecture
- `train.py` — Model training
- `evaluate.py` — Model evaluation
- `run_experiments.py` — Experiment automation
- `requirements.txt` — Python dependencies

## Setup

Create and activate a virtual environment, then install:

```bash
python -m pip install -r requirements.txt
```

## Run

Execute the scripts according to their dependencies.
Start with `train.py`, followed by `evaluate.py`, if the
scripts use that workflow.

## Notes

Generated datasets, local environments and trained model
checkpoints are excluded from Git.

This repository contains the baseline model. The research
paper-based implementation and its results will be documented
separately.
