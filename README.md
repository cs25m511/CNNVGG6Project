# VGG6 on CIFAR-10: Systems Engineering for Deep Learning – Assignment 1

**Student Name:** Jitender Kumar (CS25M511)

**Email:** cs25m511@smail.iitm.ac.in

**Program:** Web-M.Tech Programme, IIT Madras

**Course:** CS6886W – Systems Engineering for Deep Learning

---

## 1. Project Overview

Welcome to my repository for Assignment 1! This project features a clean, modular PyTorch implementation of a custom **VGG6 Convolutional Neural Network (CNN)** built to classify images from the **CIFAR-10** dataset.

The primary goal of this work is to explore how different choices—such as data normalization, data augmentation, activation functions, optimizers, learning rates, batch sizes, and batch normalization—impact deep learning model performance.

The project workflow is structured sequentially:

* **Q1:** Data preprocessing and normalization experiments.
* **Q2:** Hyperparameter exploration sweeps.
* **Q3:** Visualizations and logging via Weights & Biases (W&B).
* **Q4:** Training, checkpointing, and evaluation of the final high-performing model configuration.

---

## 2. Repository Directory Structure

Here is an overview of how the codebase is organized:

```text
Project/
├── BestModel/
│   ├── best_model.pt
├── CIFAR10Dataset/
│   ├── cifar-10-batches-py/
│   └── cifar10_channel_stats.pt
├── Control/
│   ├── __init__.py
│   └── Control.py
├── DataManager/
│   ├── __init__.py
│   ├── ArgConfig.py
│   ├── DatasetNormalizer.py
│   └── Plotting.py
├── Models/
│   ├── __init__.py
│   ├── NeuralNet.py
│   ├── Optimizers.py
│   └── Trainer.py
├── WandbLogger/
│   ├── __init__.py
│   └── WandbLogger.py
├── Q1Runs.py
├── Q2Runs.py
├── Q4Eval.py
├── Q4Train.py
└── README.md

```

### Component Breakdown

| Component | Description |
| --- | --- |
| **`BestModel/`** | Contains the saved model checkpoint (`best_model.pt`) and final experiment metadata. |
| **`CIFAR10Dataset/`** | Stores the raw CIFAR-10 data files and cached pre-computed channel normalization statistics. |
| **`Control/`** | Manages experiment execution flow, device placement, and seeding (`Control.py`). |
| **`DataManager/`** | Handles command-line arguments (`ArgConfig.py`), dataset splitting, data augmentation transforms (`DatasetNormalizer.py`), and curve plotting (`Plotting.py`). |
| **`Models/`** | Houses the VGG6 CNN architecture (`NeuralNet.py`), the optimizer creation factory (`Optimizers.py`), and the core training/evaluation loops (`Trainer.py`). |
| **`outputs/`** | Directory where training session artifacts are saved during execution. |
| **`WandbLogger/`** | Seamlessly integrates Weights & Biases for metric tracking and artifact logging. |
| **`Q1Runs.py`** | Script executing the Question 1 preprocessing comparison test cases. |
| **`Q2Runs.py`** | Script executing the Question 2 broader hyperparameter search sweeps. |
| **`Q4Train.py`** | Training script dedicated to the final optimal model configuration. |
| **`Q4Eval.py`** | Standalone script to load the saved checkpoint and run a final test evaluation. |

---

## 3. Environment and Setup

### Requirements

* Python 3.12.3
* PyTorch & Torchvision
* NumPy, Pandas, Matplotlib
* PyYAML
* Weights & Biases (`wandb`)
* A CUDA-enabled GPU is strongly recommended for faster training times.

### Installation & Virtual Environment Setup

Set up your local workspace by running the following commands in your terminal:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install required dependencies
pip install torch torchvision numpy pandas matplotlib pyyaml wandb

```

---

## 4. Dataset Preparation

The **CIFAR-10** dataset consists of 60,000 $32 \times 32$ color images distributed evenly across 10 distinct classes.

| Subset | Number of Images |
| --- | --- |
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |
| **Total** | **60,000** |

To ensure robust validation, a deterministic random seed (`seed = 19`) is used to split the training set into the 45k/5k train/validation subsets.

### Normalization & Caching

To speed up data loaders, per-channel mean and standard deviation are calculated from the training split and automatically cached locally in:

```text
CIFAR10Dataset/cifar10_channel_stats.pt

```

### Data Augmentation Strategy

The training pipeline incorporates regularizations to prevent overfitting:

* **Random Crop:** Applied with a padding of 4 pixels.
* **Random Horizontal Flip:** Applied with a $0.5$ probability.
* **Color Jitter:** Optional appearance augmentation.
* **Random Erasing:** Optional occlusion-based regularization.

*Note: Validation and test sets are strictly evaluated using tensor conversion and channel normalization, excluding any stochastic training augmentations.*

---

## 5. Summary of Experiments

### 5.1 Question 1 – Training Baseline and Preprocessing

Five primary preprocessing configurations (**TC1–TC5**) were evaluated to find the best data pipeline structure:

* **TC1:** Manual normalization ($\mu = \sigma = 0.5$), no jitter, no random erasing.
* **TC2:** Dataset-derived channel statistics, no jitter, no random erasing.
* **TC3:** Dataset statistics + Color Jitter enabled.
* **TC4:** Dataset statistics + Random Erasing enabled.
* **TC5:** Dataset statistics + both Color Jitter and Random Erasing enabled.

These preprocessing setups were crossed with 5 different activation functions (ReLU, Sigmoid, Tanh, SiLU, GELU), resulting in a comprehensive 25-run grid search.

### 5.2 Question 2 – Model Performance Across Configurations

Exploratory sweeps were conducted to study the sensitivity of the network against:

* **Activation Functions:** ReLU, Sigmoid, Tanh, SiLU, GELU.
* **Optimizers:** SGD, Nesterov-SGD, Adam, Adagrad, RMSprop, Nadam.
* **Hyperparameters:** Batch size scaling, learning rate tuning, total training epochs, and the inclusion of Batch Normalization layers.

### 5.3 Question 3 – W&B Visualizations

Weights & Biases dashboards were leveraged to automatically extract and display:

* Parallel-coordinate hyperparameter maps.
* Validation accuracy step scatter graphs.
* Epoch-wise training loss, training accuracy, validation loss, and validation accuracy tracks.

### 5.4 Question 4 – Final Model Performance

Based on insights gathered from the W&B tracking sweeps, **Run 22** emerged as the winning recipe, combining a refined optimizer choice, Batch Normalization, and extended training iterations:

| Hyperparameter | Value |
| --- | --- |
| Activation Function | ReLU |
| Optimizer | Nadam |
| Batch Size | 128 |
| Epochs | 120 |
| Learning Rate | 0.001 |
| Batch Normalization | Enabled (`use_batchnorm = True`) |
| Weight Decay | 0.0001 |
| Random Seed | 19 |
| Normalization | Dataset-derived channel statistics |

**Performance Metrics:**

* **Best Validation Accuracy:** 87.10%
* **Test Accuracy:** 86.90%
* **Test Loss:** 0.3916

*(Note: Minor performance deltas can naturally occur across separate hardware platforms due to hardware-level nondeterministic GPU operations.)*

---

## 6. Execution Instructions

Execute all scripts directly from the project root directory.

### 6.1 Question 1 – Preprocessing Experiments

```bash
python Q1Runs.py --wandb

```

### 6.2 Question 2 – Hyperparameter Sweeps

```bash
python Q2Runs.py --groups all --wandb

```

### 6.3 Question 4 – Train the Final Optimal Model

To run the winning configuration and persist its artifacts:

```bash
python Q4Train.py

```

This script initializes the CIFAR-10 data loaders, instantiates the VGG6 network with Batch Normalization, trains for 120 epochs while monitoring validation metrics, and outputs the model weights to:

```text
outputs/
├── Q4_BestModel/
    ├── best_model.pt
    ├── history.csv
    └── result.json

```

### 6.4 Question 4 – Evaluate the Saved Checkpoint

To independently test the saved weights against the unseen test partition:

```bash
python Q4Eval.py

```

This script loads `BestModel/best_model.pt` (or your output path), builds the evaluation data pipeline, and outputs the final test loss and top-1 accuracy.

---

## 7. Model Reproducibility Guidelines

To replicate these exact reported scores, ensure the following parameters remain unchanged:

* VGG6 network layer layout and dimensions.
* Global random seed setting (`seed = 19`).
* Dataset partitioning rules (45k/5k/10k splits).
* Preprocessing normalization constants and augmentation transforms.
* Chosen activation functions, optimizers, learning rate schedules, and batch normalization states.

---

## 8. Summary of Submission Artifacts

| Artifact | Location / Purpose |
| --- | --- |
| **Source Code** | Full modular implementation (`Control/`, `DataManager/`, `Models/`, etc.) |
| **Model Weights** | `BestModel/best_model.pt` |
| **Training History** | `BestModel/history.csv` (Epoch-level progress metrics) |
| **Metrics Summary** | `BestModel/result.json` |
| **Visualizations** | W&B tracking graphs demonstrating convergence and loss behavior |
| **Documentation** | This `README.md` file |