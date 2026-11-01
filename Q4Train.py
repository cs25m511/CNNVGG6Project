"""
Q4Train.py: Training script for the best performing VGG6 configuration on CIFAR-10.
Achieved Best Validation Accuracy: 87.10% | Test Accuracy: 86.90%
Configuration: ReLU activation, Nadam optimizer, Batch Normalization enabled,
               Batch Size: 128, Epochs: 120, Learning Rate: 0.001, Seed: 19.
"""

from pathlib import Path
import torch
import pandas as pd

from DataManager.DatasetNormalizer import DatasetNormalizer
from Models.NeuralNet import NeuralNet
from Models.Optimizers import OptimizerFactory
from Models.Trainer import Trainer


def main():
    config_dict = {
        "activation": "relu",
        "optimizer": "nadam",
        "batch_size": 128,
        "epochs": 120,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "use_batchnorm": True,
        "seed": 19,
        "num_workers": 2,
        "device": "auto",
        "data_dir": "./CIFAR10Dataset",
        "output_dir": "./outputs/Q4_BestModel",
        "calc_meanstd": True,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
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
        "evaluate_test": True,
    }

    output_dir = Path(config_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for complete reproducibility
    torch.manual_seed(config_dict["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config_dict["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and config_dict["device"] != "cpu" else "cpu")
    print(f"Using device: {device}")

    # Build reproducible data loaders using DatasetNormalizer class
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

    # Initialize Neural Net Model with Batch Normalization
    model = NeuralNet(
        activation=config_dict["activation"],
        use_batchnorm=config_dict["use_batchnorm"],
    ).to(device)

    # Create Optimizer via OptimizerFactory
    optimizer = OptimizerFactory.create(
        name=config_dict["optimizer"],
        model_parameters=model.parameters(),
        learning_rate=config_dict["learning_rate"],
        momentum=config_dict["momentum"],
        weight_decay=config_dict["weight_decay"],
    )

    # Initialize Trainer without logger
    trainer = Trainer(model, optimizer, device, output_dir, logger=None)

    print("\nStarting training with the optimal configuration...")
    history = trainer.train(train_loader, val_loader, config_dict["epochs"], config_dict)

    # Save epoch-wise history to CSV
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    # Evaluate on Held-out Test Set
    print("\nEvaluating best model checkpoint on test set...")
    test_metrics = trainer.evaluate_test(test_loader)

    print("=" * 60)
    print("Training Complete Successfully!")
    print(f"Best Validation Accuracy: {trainer.best_validation_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_metrics['test/loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['test/accuracy'] * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()