import argparse
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class ArgConfig:
    # Model and training hyperparameters
    activation: str
    optimizer: str
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    use_batchnorm: bool

    # General settings
    seed: int
    num_workers: int
    device: str
    data_dir: str
    output_dir: str

    # Dataset normalization
    calc_meanstd: bool
    normalize_mean: Tuple[float, float, float]
    normalize_std: Tuple[float, float, float]

    # Data augmentation
    add_crop: bool
    crop_padding: int
    horizontal_flip_prob: float

    add_jitter: bool
    jitter_brightness: float
    jitter_contrast: float
    jitter_saturation: float
    jitter_hue: float

    add_random_erasing: bool
    random_erasing_prob: float
    random_erasing_scale: Tuple[float, float]
    random_erasing_ratio: Tuple[float, float]

    # W&B settings
    wandb_project: str
    wandb_entity: Optional[str]
    run_name: Optional[str]
    no_wandb: bool
    no_download: bool

    @classmethod
    def from_cli(cls) -> "ArgConfig":
        parser = argparse.ArgumentParser(description="Train a CNN model on CIFAR-10")

        # --------------------------------------------------
        # Model and training hyperparameters
        # --------------------------------------------------
        parser.add_argument("--activation", default="relu", choices=["relu", "sigmoid", "tanh", "silu", "gelu"])
        parser.add_argument("--optimizer", default="sgd",
                            choices=["sgd", "nesterov_sgd", "adam", "adagrad", "rmsprop", "nadam"])
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--learning_rate", type=float, default=0.01)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--use_batchnorm", action="store_true", help="Enable Batch Normalization in the model")

        # --------------------------------------------------
        # General settings
        # --------------------------------------------------
        parser.add_argument("--seed", type=int, default=19)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        parser.add_argument("--data_dir", default="./CIFAR10Dataset")
        parser.add_argument("--output_dir", default="./outputs")

        # --------------------------------------------------
        # Dataset normalization
        # --------------------------------------------------
        parser.add_argument("--calc_meanstd", action=argparse.BooleanOptionalAction, default=True,
                            help="Calculate/load CIFAR-10 channel statistics. Use --no-calc_meanstd to use the supplied mean/std.")
        parser.add_argument("--normalize_mean", type=float, nargs=3, default=(0.5, 0.5, 0.5), metavar=("R", "G", "B"),
                            help="Fallback normalization means for the RGB channels")
        parser.add_argument("--normalize_std", type=float, nargs=3, default=(0.5, 0.5, 0.5), metavar=("R", "G", "B"),
                            help="Fallback normalization standard deviations for RGB")

        # --------------------------------------------------
        # Random crop and horizontal flip
        # --------------------------------------------------
        parser.add_argument("--add_crop", action=argparse.BooleanOptionalAction, default=True,
                            help="Enable/disable random cropping")
        parser.add_argument("--crop_padding", type=int, default=4, help="Padding for RandomCrop")
        parser.add_argument("--horizontal_flip_prob", type=float, default=0.5,
                            help="Probability of random horizontal flipping")

        # --------------------------------------------------
        # Color jitter
        # --------------------------------------------------
        parser.add_argument("--add_jitter", action=argparse.BooleanOptionalAction, default=False,
                            help="Enable/disable ColorJitter")
        parser.add_argument("--jitter_brightness", type=float, default=0.15)
        parser.add_argument("--jitter_contrast", type=float, default=0.15)
        parser.add_argument("--jitter_saturation", type=float, default=0.15)
        parser.add_argument("--jitter_hue", type=float, default=0.02)

        # --------------------------------------------------
        # Random erasing
        # --------------------------------------------------
        parser.add_argument("--add_random_erasing", action=argparse.BooleanOptionalAction, default=True,
                            help="Enable/disable random erasing")
        parser.add_argument("--random_erasing_prob", type=float, default=0.25)
        parser.add_argument("--random_erasing_scale", type=float, nargs=2, default=(0.02, 0.10), metavar=("MIN", "MAX"))
        parser.add_argument("--random_erasing_ratio", type=float, nargs=2, default=(0.5, 2.0), metavar=("MIN", "MAX"))

        # --------------------------------------------------
        # W&B settings
        # --------------------------------------------------
        parser.add_argument("--wandb_project", default="Normalization")
        parser.add_argument("--wandb_entity", default=None)
        parser.add_argument("--run_name", default=None)
        parser.add_argument("--no_wandb", action="store_true", help="Disable W&B tracking")
        parser.add_argument("--no_download", action="store_true", help="Do not download CIFAR-10")

        args = parser.parse_args()
        config = cls(**vars(args))
        config.validate()
        return config

    def validate(self) -> None:
        """Validate configuration values before training."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3:
            raise ValueError("Normalization mean and std must have 3 values.")
        if any(value <= 0 for value in self.normalize_std):
            raise ValueError("All normalization std values must be positive.")
        if self.crop_padding < 0:
            raise ValueError("crop_padding cannot be negative.")
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be between 0 and 1.")
        if not 0.0 <= self.random_erasing_prob <= 1.0:
            raise ValueError("random_erasing_prob must be between 0 and 1.")

        for name, bounds in [("random_erasing_scale", self.random_erasing_scale),
                             ("random_erasing_ratio", self.random_erasing_ratio)]:
            if bounds[0] <= 0 or bounds[1] < bounds[0]:
                raise ValueError(f"Invalid {name}: {bounds}")

        if not 0.0 <= self.jitter_hue <= 0.5:
            raise ValueError("jitter_hue must be between 0 and 0.5.")

        for name in ("jitter_brightness", "jitter_contrast", "jitter_saturation"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")
        setattr(self, key, value)