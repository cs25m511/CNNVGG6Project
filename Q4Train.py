
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from DataManager.DatasetNormalizer import DatasetNormalizer
from Models.NeuralNet import NeuralNet
from Models.Optimizers import OptimizerFactory
from Models.Trainer import Trainer


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU and CUDA random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic kernels and disable autotuning.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Some operations may not have deterministic implementations on every
    # installed backend. warn_only=True reports these without aborting training.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Compatibility with older PyTorch releases.
        torch.use_deterministic_algorithms(True)

    # Avoid TF32 differences on supported NVIDIA GPUs.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def main() -> None:
    config_dict = {
        # Run 22: selected best-validation configuration
        "activation": "relu",
        "optimizer": "nadam",
        "batch_size": 128,
        "epochs": 120,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "use_batchnorm": True,
        "seed": 19,

        # Runtime and paths
        "num_workers": 2,
        "device": "auto",  # auto, cpu, or cuda
        "data_dir": "./CIFAR10Dataset",
        # Keep this aligned with Q4Eval.py and the README.
        "output_dir": "./BestModel",

        # Preprocessing: use the cached/calculated CIFAR-10 channel statistics.
        "calc_meanstd": True,
        "normalize_mean": (0.5, 0.5, 0.5),  # ignored when calc_meanstd=True
        "normalize_std": (0.5, 0.5, 0.5),   # ignored when calc_meanstd=True
        "add_crop": True,
        "crop_padding": 4,
        "horizontal_flip_prob": 0.5,
        "add_jitter": False,
        "jitter_brightness": 0.15,
        "jitter_contrast": 0.15,
        "jitter_saturation": 0.15,
        "jitter_hue": 0.02,
        "add_random_erasing": False,
        "random_erasing_prob": 0.25,
        "random_erasing_scale": (0.02, 0.10),
        "random_erasing_ratio": (0.5, 2.0),
        "no_download": False,
    }

    # Seed before constructing the dataset, model, optimizer, or data loaders.
    set_seed(config_dict["seed"])

    if config_dict["device"] == "cpu":
        device = torch.device("cpu")
    elif config_dict["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested, but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(config_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Q4 – Final CIFAR-10 Model Training")
    print(f"Seed: {config_dict['seed']}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir.resolve()}")
    print("Deterministic algorithms: enabled (warn_only=True)")
    print("=" * 72)

    # DatasetNormalizer uses the supplied seed for the 45k/5k split and the
    # training DataLoader shuffle generator. Validation/test transforms should
    # remain deterministic and must not include training augmentation.
    data_module = DatasetNormalizer(
        data_dir=config_dict["data_dir"],
        batch_size=config_dict["batch_size"],
        num_workers=config_dict["num_workers"],
        seed=config_dict["seed"],
        calc_meanstd=config_dict["calc_meanstd"],
        normalize_mean=config_dict["normalize_mean"],
        normalize_std=config_dict["normalize_std"],
        add_crop=config_dict["add_crop"],
        crop_padding=config_dict["crop_padding"],
        horizontal_flip_prob=config_dict["horizontal_flip_prob"],
        add_jitter=config_dict["add_jitter"],
        add_random_erasing=config_dict["add_random_erasing"],
        download=not config_dict["no_download"],
    )
    train_loader, val_loader, test_loader = data_module.build_loaders()

    # Re-seed immediately before model creation so model initialization starts
    # from the same RNG state even if dataset/statistics setup used randomness.
    set_seed(config_dict["seed"])

    model = NeuralNet(
        activation=config_dict["activation"],
        use_batchnorm=config_dict["use_batchnorm"],
    ).to(device)

    optimizer = OptimizerFactory.create(
        name=config_dict["optimizer"],
        model_parameters=model.parameters(),
        learning_rate=config_dict["learning_rate"],
        momentum=config_dict["momentum"],
        weight_decay=config_dict["weight_decay"],
    )

    trainer = Trainer(model, optimizer, device, output_dir, logger=None)

    print("\nStarting training with the selected Run 22 configuration...\n")
    history = trainer.train(
        train_loader,
        val_loader,
        config_dict["epochs"],
        config_dict,
    )

    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Trainer.evaluate_test() restores the checkpoint with the highest
    # validation accuracy before computing held-out test metrics.
    print("\nEvaluating the best-validation checkpoint on the held-out test set...")
    test_metrics = trainer.evaluate_test(test_loader)

    best_checkpoint_path = output_dir / "best_model.pt"
    checkpoint = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    best_epoch = int(checkpoint.get("epoch", -1))
    best_val_accuracy = float(checkpoint["best_validation_accuracy"])

    results = {
        "configuration": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in config_dict.items()
        },
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": float(test_metrics["test/loss"]),
        "test_accuracy": float(test_metrics["test/accuracy"]),
        "checkpoint": str(best_checkpoint_path),
        "history_csv": str(history_path),
    }

    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 72)
    print("Training completed.")
    print(f"Best validation epoch : {best_epoch}")
    print(f"Best validation acc.  : {best_val_accuracy * 100:.2f}%")
    print(f"Test loss             : {test_metrics['test/loss']:.4f}")
    print(f"Test accuracy         : {test_metrics['test/accuracy'] * 100:.2f}%")
    print(f"Checkpoint            : {best_checkpoint_path}")
    print(f"History               : {history_path}")
    print(f"Results               : {result_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
