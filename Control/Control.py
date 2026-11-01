import torch
from pathlib import Path
import json
import pandas as pd
import os
import random
import numpy as np


from WandbLogger import WandbLogger
from DataManager import DatasetNormalizer
from Models.NeuralNet import NeuralNet
from Models.Optimizers import OptimizerFactory
from Models.Trainer import Trainer


class Control:
    def __init__(self):
        self.config_m = None
        self.output_root = Path("./outputs")

    def setconfig(self, config):
        self.config_m = config
        self.output_root = Path(config.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _setseed(self, seed: int):
        """Seed common random-number generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in PyTorch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Environment variable for CUDA cuBLAS reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def _get_device(self):
        if self.config_m.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but is not available.")
            return torch.device("cuda")
        elif self.config_m.device == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def start(self):
        if self.config_m is None:
            raise RuntimeError(
                "Configuration is missing. Call setconfig() first."
            )

        seed = self.config_m.seed

        # Seed before creating the model or data loaders.
        self._setseed(seed)
        device = self._get_device()

        activation = self.config_m.activation
        optimizer_name = self.config_m.optimizer
        batch_size = self.config_m.batch_size
        epochs = self.config_m.epochs
        learning_rate = self.config_m.learning_rate
        momentum = self.config_m.momentum
        weight_decay = self.config_m.weight_decay
        use_batchnorm = self.config_m.use_batchnorm

        config = {
            "activation": activation,
            "optimizer": optimizer_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "use_batchnorm": use_batchnorm,
            "seed": seed,
            "device": str(device),
        }

        # Give each configuration its own output directory.
        bn_str = "bn" if use_batchnorm else "nobn"

        run_label = (
            f"{activation}_{optimizer_name}_{bn_str}"
            f"_bs{batch_size}_ep{epochs}_lr{learning_rate}"
            f"_seed{seed}"
        )

        safe_label = (
            run_label.replace("/", "_")
            .replace(" ", "_")
        )

        run_dir = self.output_root / safe_label
        run_dir.mkdir(parents=True, exist_ok=True)

        logger = WandbLogger.WandbLogger(
            enabled=not self.config_m.no_wandb,
            project=self.config_m.wandb_project,
            entity=self.config_m.wandb_entity,
            config=config,
            name=run_label,
        )

        try:
            # 1. Load training, validation, and test loaders.
            data_module = DatasetNormalizer.DatasetNormalizer(
                data_dir=self.config_m.data_dir,
                batch_size=batch_size,
                num_workers=self.config_m.num_workers,
                seed=seed,
                download=not self.config_m.no_download,

                # Normalization
                calc_meanstd=self.config_m.calc_meanstd,
                normalize_mean=self.config_m.normalize_mean,
                normalize_std=self.config_m.normalize_std,

                # Crop and horizontal flip
                add_crop=self.config_m.add_crop,
                crop_padding=self.config_m.crop_padding,
                horizontal_flip_prob=self.config_m.horizontal_flip_prob,

                # Color jitter
                add_jitter=self.config_m.add_jitter,
                jitter_brightness=self.config_m.jitter_brightness,
                jitter_contrast=self.config_m.jitter_contrast,
                jitter_saturation=self.config_m.jitter_saturation,
                jitter_hue=self.config_m.jitter_hue,

                # Random erasing
                add_random_erasing=self.config_m.add_random_erasing,
                random_erasing_prob=self.config_m.random_erasing_prob,
                random_erasing_scale=self.config_m.random_erasing_scale,
                random_erasing_ratio=self.config_m.random_erasing_ratio,
            )

            train_loader, val_loader, test_loader = data_module.build_loaders()

            # 2. Create a fresh model for this experiment.
            model = NeuralNet(
                activation=activation,
                use_batchnorm=use_batchnorm,
            ).to(device)

            # 3. Create the optimizer.
            optimizer_instance = OptimizerFactory.create(
                optimizer_name,
                model.parameters(),
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            # 4. Train and evaluate on the validation set.
            trainer = Trainer(model, optimizer_instance, device, run_dir, logger)
            history = trainer.train(train_loader, val_loader, epochs, config)

            # Evaluate the best validation checkpoint on the test dataset
            test_metrics = trainer.evaluate_test(test_loader)

            # Display test results
            print("\nFinal Test Results")
            print("------------------")
            print(f"Test Loss: {test_metrics['test/loss']:.4f}")
            print(
                f"Test Accuracy: "
                f"{test_metrics['test/accuracy'] * 100:.2f}%"
            )

            # Log test metrics to W&B
            if logger is not None:
                logger.log_epoch(test_metrics)

            # 5. Save epoch-wise history.
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            # 6. Test evaluation is opt-in.
            # Keep this False for hyperparameter sweeps.
            run_test_evaluation = getattr(
                self.config_m,
                "evaluate_test",
                False,
            )

            test_metrics = {}

            if run_test_evaluation:
                test_metrics = trainer.evaluate_test(test_loader)

            # 7. Log the best validation result.
            summary = {
                "best_validation_accuracy":
                    trainer.best_validation_accuracy,
            }
            summary.update(test_metrics)

            logger.log_summary(summary)

            # Log the best-validation checkpoint.
            if trainer.best_checkpoint.exists():
                logger.log_model(trainer.best_checkpoint)

            # 8. Save machine-readable results.
            result = {
                **config,
                "run_name": run_label,
                "best_validation_accuracy":
                    trainer.best_validation_accuracy,
                "checkpoint": str(trainer.best_checkpoint),
            }
            result.update(test_metrics)

            with open(
                    run_dir / "result.json",
                    "w",
                    encoding="utf-8",
            ) as result_file:
                json.dump(result, result_file, indent=2)

            message = (
                f"Run complete: {run_label} | "
                f"Best validation accuracy: "
                f"{trainer.best_validation_accuracy:.4f}"
            )

            if "test/accuracy" in test_metrics:
                message += (
                    f" | Test accuracy: "
                    f"{test_metrics['test/accuracy']:.4f}"
                )

            print(message)
            return result

        finally:
            # Close the W&B run even if training raises an exception.
            logger.finish()