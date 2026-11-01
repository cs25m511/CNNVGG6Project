from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class DatasetNormalizer:
    def __init__(
        self,
        data_dir="./CIFAR10Dataset",
        batch_size=32,
        num_workers=2,
        seed=42,
        calc_meanstd=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        add_crop=True,
        crop_padding=4,
        horizontal_flip_prob=0.5,
        add_jitter=False,
        jitter_brightness=0.15,
        jitter_contrast=0.15,
        jitter_saturation=0.15,
        jitter_hue=0.02,
        add_random_erasing=True,
        random_erasing_prob=0.25,
        random_erasing_scale=(0.02, 0.10),
        random_erasing_ratio=(0.5, 2.0),
        download=False,
        force_recalculate_stats=False,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.download = download

        self.calc_meanstd = calc_meanstd
        self.normalize_mean = tuple(normalize_mean)
        self.normalize_std = tuple(normalize_std)
        self.force_recalculate_stats = force_recalculate_stats

        if len(self.normalize_mean) != 3:
            raise ValueError("normalize_mean must contain exactly 3 values.")

        if len(self.normalize_std) != 3:
            raise ValueError("normalize_std must contain exactly 3 values.")

        if any(s <= 0 for s in self.normalize_std):
            raise ValueError("All normalize_std values must be positive.")

        if not 0 <= horizontal_flip_prob <= 1:
            raise ValueError("horizontal_flip_prob must be between 0 and 1.")

        if not 0 <= random_erasing_prob <= 1:
            raise ValueError("random_erasing_prob must be between 0 and 1.")

        self.add_crop = add_crop
        self.crop_padding = crop_padding
        self.horizontal_flip_prob = horizontal_flip_prob

        self.add_jitter = add_jitter
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.jitter_saturation = jitter_saturation
        self.jitter_hue = jitter_hue

        self.add_random_erasing = add_random_erasing
        self.random_erasing_prob = random_erasing_prob
        self.random_erasing_scale = tuple(random_erasing_scale)
        self.random_erasing_ratio = tuple(random_erasing_ratio)

        self.stats_file = self.data_dir / "cifar10_channel_stats.pt"

        # These are assigned when _transforms() is called.
        self.mean = None
        self.std = None

    def _calculate_mean_std(self):
        """Calculate per-channel mean and population standard deviation."""

        dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=self.download,
            transform=transforms.ToTensor(),
        )

        loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        channel_sum = torch.zeros(3, dtype=torch.float64)
        channel_squared_sum = torch.zeros(3, dtype=torch.float64)
        total_pixels = 0

        for images, _ in loader:
            images = images.to(torch.float64)

            batch_pixels = (
                images.shape[0] * images.shape[2] * images.shape[3]
            )

            channel_sum += images.sum(dim=(0, 2, 3))
            channel_squared_sum += images.square().sum(dim=(0, 2, 3))

            total_pixels += batch_pixels

        mean = channel_sum / total_pixels

        variance = (
            channel_squared_sum / total_pixels
        ) - mean.square()

        std = torch.sqrt(variance.clamp(min=0.0))

        return mean.float(), std.float()

    def _get_mean_std(self):
        """Load valid cached statistics or calculate and save them."""

        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

        # Force recalculation when requested.
        if (
            self.stats_file.exists()
            and not self.force_recalculate_stats
        ):
            try:
                stats = torch.load(
                    self.stats_file,
                    map_location="cpu",
                    weights_only=True,
                )

                mean = torch.as_tensor(
                    stats["mean"], dtype=torch.float32
                ).reshape(-1)

                std = torch.as_tensor(
                    stats["std"], dtype=torch.float32
                ).reshape(-1)

                valid = (
                    mean.numel() == 3
                    and std.numel() == 3
                    and torch.isfinite(mean).all().item()
                    and torch.isfinite(std).all().item()
                    and (std > 0).all().item()
                )

                if valid:
                    print(
                        f"\nLoaded cached CIFAR-10 statistics:\n"
                        f"File: {self.stats_file}\n"
                        f"Mean: {tuple(mean.tolist())}\n"
                        f"Std:  {tuple(std.tolist())}"
                    )

                    return tuple(mean.tolist()), tuple(std.tolist())

                print(
                    "Cached statistics are invalid. "
                    "Recalculating..."
                )

            except (
                OSError,
                RuntimeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                print(
                    f"Could not load cached statistics: {exc}\n"
                    "Recalculating..."
                )

        print("\nCalculating CIFAR-10 training-set statistics...")

        mean_tensor, std_tensor = self._calculate_mean_std()

        mean = tuple(mean_tensor.tolist())
        std = tuple(std_tensor.tolist())

        # Save calculated statistics for future runs.
        torch.save(
            {
                "mean": mean_tensor,
                "std": std_tensor,
            },
            self.stats_file,
        )

        print("Statistics calculated successfully.")
        print(f"Saved to: {self.stats_file}")
        print(f"Calculated dataset mean: {mean}")
        print(f"Calculated dataset std:  {std}")

        return mean, std

    def _transforms(self):
        """Create training and evaluation transforms."""

        if self.calc_meanstd:
            # Calculate or load dataset statistics.
            mean, std = self._get_mean_std()

            print("\nNormalization mode: DATASET STATISTICS")

        else:
            # Use manually supplied normalization values.
            mean = self.normalize_mean
            std = self.normalize_std

            print("\nNormalization mode: MANUAL VALUES")

        # Save the final selected values for inspection.
        self.mean = tuple(mean)
        self.std = tuple(std)

        print(f"Final normalization mean: {self.mean}")
        print(f"Final normalization std:  {self.std}")

        train_transforms = []

        # 1. Random crop.
        if self.add_crop:
            train_transforms.append(
                transforms.RandomCrop(
                    32,
                    padding=self.crop_padding,
                )
            )

        # 2. Random horizontal flip.
        if self.horizontal_flip_prob > 0:
            train_transforms.append(
                transforms.RandomHorizontalFlip(
                    p=self.horizontal_flip_prob
                )
            )

        # 3. Color jitter.
        if self.add_jitter:
            train_transforms.append(
                transforms.ColorJitter(
                    brightness=self.jitter_brightness,
                    contrast=self.jitter_contrast,
                    saturation=self.jitter_saturation,
                    hue=self.jitter_hue,
                )
            )

        # 4. Convert PIL images to tensors in [0, 1].
        train_transforms.append(transforms.ToTensor())

        # 5. Normalize using the selected mean and std.
        train_transforms.append(
            transforms.Normalize(
                mean=self.mean,
                std=self.std,
            )
        )

        # 6. Random erasing operates on normalized tensors.
        if self.add_random_erasing:
            train_transforms.append(
                transforms.RandomErasing(
                    p=self.random_erasing_prob,
                    scale=self.random_erasing_scale,
                    ratio=self.random_erasing_ratio,
                    value=0,
                )
            )

        train_transform = transforms.Compose(train_transforms)

        # Validation and test: no random augmentation.
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

        return train_transform, eval_transform

    def build_loaders(self):
        """Build reproducible training, validation, and test loaders."""

        train_transform, eval_transform = self._transforms()

        # Training data uses the training augmentation pipeline.
        train_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=self.download,
            transform=train_transform,
        )

        # Separate dataset instance for validation.
        eval_train_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=self.download,
            transform=eval_transform,
        )

        # Test data has no random augmentation.
        test_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=False,
            download=self.download,
            transform=eval_transform,
        )

        # Reproducible 45,000 / 5,000 train-validation split.
        generator = torch.Generator().manual_seed(self.seed)

        indices = torch.randperm(
            len(train_dataset),
            generator=generator,
        ).tolist()

        val_indices = indices[:5000]
        train_indices = indices[5000:]

        train_set = Subset(train_dataset, train_indices)
        val_set = Subset(eval_train_dataset, val_indices)

        common_options = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }

        loader_generator = torch.Generator().manual_seed(self.seed)

        train_loader = DataLoader(
            train_set,
            shuffle=True,
            generator=loader_generator,
            **common_options,
        )

        val_loader = DataLoader(
            val_set,
            shuffle=False,
            **common_options,
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **common_options,
        )

        print(f"\nTraining images:   {len(train_set)}")
        print(f"Validation images: {len(val_set)}")
        print(f"Test images:       {len(test_dataset)}")

        print(f"Normalization mean: {self.mean}")
        print(f"Normalization std:  {self.std}")

        return train_loader, val_loader, test_loader