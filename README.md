# Aircraft Damage Classifier

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)](https://keras.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-purple?logo=huggingface)](https://huggingface.co/transformers/)

## Description

`Aircraft Damage Classifier` is a proof-of-concept binary image classification pipeline implemented in Python using TensorFlow/Keras. The project is focused on distinguishing between damaged and undamaged aircraft surface images while documenting an iterative model debugging process. The code demonstrates a transition from a simple baseline model to a more robust architecture with dropout regularization and early stopping to prevent memorization.

## Overview

- **Task:** Binary image classification (2 classes)
- **Framework:** TensorFlow / Keras
- **Dataset:** `aircraft_damage_dataset_v1` — 300 training images, 96 validation images, 50 test images
- **Architecture:** VGG16-based feature extractor + dense classifier head
- **Key techniques:** Rescaling, Dropout, EarlyStopping, transfer learning with frozen pretrained backbone
- **Extras:** Dataset download and extraction are handled automatically by the script

## Project Structure

```text
keras_proj/
├── aircraft_damage_classifier.py   # Main training script
├── README.md
└── .venv/                          # Optional virtual environment folder
```

## Requirements

The following packages are required to run the project successfully:

- `tensorflow` (or `tensorflow-cpu` for CPU-only environments)
- `tf-keras`
- `numpy`
- `matplotlib`
- `Pillow`
- `torch`
- `transformers`

> Note: The current implementation imports `tf_keras` explicitly and also uses PyTorch/Transformers for the BLIP captioning portion.

## Architecture

The model is built in two main stages:

1. **Feature extraction**
   - Uses a pretrained `VGG16` backbone from `tf_keras.applications` with `include_top=False`
   - The backbone is frozen so only the classifier head is trained
   - Backbone output is flattened into a feature vector

2. **Classification head**
   - Dense layer with 512 units and ReLU activation
   - Dropout with `rate=0.3`
   - Second Dense layer with 512 units and ReLU activation
   - Another Dropout layer with `rate=0.3`
   - Final sigmoid output for binary classification

The training loop includes an `EarlyStopping` callback that monitors validation loss and restores the best weights when the metric stops improving.

## Installation

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install tensorflow tf-keras numpy matplotlib Pillow torch transformers
```

4. Run the main training script:

```powershell
python aircraft_damage_classifier.py
```

## Usage

Running `aircraft_damage_classifier.py` will:

- download and extract `aircraft_damage_dataset_v1`
- create image generators for train/validation/test sets
- build a VGG16-based classifier
- train the model with early stopping
- plot training loss, validation loss, and accuracy curves
- evaluate the model on the test split
- visualize a sample prediction from the test set

## Demo

> Add screenshots, GIFs, or a short video here demonstrating the training curves and sample predictions.

## Model Iteration & Overfitting Analysis

This project documents an iterative debugging process that moved from an underfit baseline to a regularized model with early stopping:

### Iteration 1 — Baseline (5 epochs, minimal regularization)

- Train accuracy: 86.3%
- Validation accuracy: 67.7%
- Validation loss: 0.51
- Issue: underfitting caused by insufficient training capacity / epochs
- <img width="801" height="690" alt="Captura de pantalla 2026-07-01 172342" src="https://github.com/user-attachments/assets/db690912-658a-48cf-bcac-c3077b666a9d" />


### Iteration 2 — Extended training (20 epochs, no strong regularization)

- Train accuracy: 99.3%
- Validation accuracy: ~78%
- Validation loss: dropped to 0.41 then rose to 0.53
- Issue: overfitting; model memorized training samples rather than generalizing
- <img width="801" height="677" alt="Captura de pantalla 2026-07-01 202453" src="https://github.com/user-attachments/assets/bbdf364b-6025-4001-85a4-1f9e63e197e5" />
<img width="796" height="681" alt="Captura de pantalla 2026-07-01 202602" src="https://github.com/user-attachments/assets/cb36241f-b090-412d-9849-644d1d2af31a" />
<img width="622" height="707" alt="Captura de pantalla 2026-07-01 202613" src="https://github.com/user-attachments/assets/c70e8cde-9bc4-466c-836f-4f4a36e90570" />



### Iteration 3 — Final model (Dropout + Early Stopping)

The final model adds dropout before dense layers and uses an `EarlyStopping` callback with:

- `monitor='val_loss'`
- `patience=3`
- `restore_best_weights=True`

The final iteration is designed to stop training once validation loss ceases to improve and to recover the best-performing weights.
<img width="802" height="691" alt="Captura de pantalla 2026-07-02 155748" src="https://github.com/user-attachments/assets/3dd0c6be-20d4-4917-9496-d99fafb6bd4e" />
<img width="621" height="623" alt="Captura de pantalla 2026-07-02 155829" src="https://github.com/user-attachments/assets/f893544f-15bd-4f1a-8179-a25e764aec59" />
<img width="788" height="591" alt="Captura de pantalla 2026-07-02 155812" src="https://github.com/user-attachments/assets/6ceb8021-fce2-4287-8378-84d50a768e6e" />

## Results

| Iteration | Train Acc | Val Acc | Val Loss | Issue |
|---|---|---|---|---|
| Baseline | 86.3% | 67.7% | 0.51 | Underfitting |
| Extended | 99.3% | ~78% | 0.53 (rising) | Overfitting |
| Regularized | ~93% | ~79% | 0.407 (best) | Better generalization |

## Key Takeaways

- More epochs alone do not guarantee better generalization.
- `Dropout` helps reduce memorization on a small dataset.
- `EarlyStopping` with `restore_best_weights=True` is an effective way to avoid manual epoch selection.
- With only 300 training images, regularization is essential for a dependable model.

## Roadmap

- [ ] Expand the dataset or apply stronger augmentation (rotation, zoom, flips)
- [ ] Try other pretrained backbones such as `MobileNetV2` or `ResNet50`
- [ ] Add a confusion matrix and classification report for test evaluation
- [ ] Experiment with L2 regularization in addition to Dropout
- [ ] Add data augmentation to the training generator for better generalization
- [ ] Separate the captioning pipeline into its own script or module

## Contribution

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch
3. Open a pull request with a clear description
4. Add tests or documentation for major changes

Please submit bug reports or improvement ideas via issues.
