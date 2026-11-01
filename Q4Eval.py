"""
Q4Eval.py: Evaluation script to load the saved best model checkpoint
and evaluate its top-1 accuracy and loss on the CIFAR-10 test set.
"""

from pathlib import Path
import torch

from DataManager.DatasetNormalizer import DatasetNormalizer
from Models.NeuralNet import NeuralNet
from Models.Trainer import Trainer


def main():
    checkpoint_path = Path("./BestModel/best_model.pt")

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}. Please run Q4Train.py first.")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    print(f"Loaded model configuration parameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using evaluation device: {device}")

    # Correctly instantiate DatasetNormalizer class for evaluation loader
    data_module = DatasetNormalizer(
        data_dir=config.get("data_dir", "./CIFAR10Dataset"),
        batch_size=config.get("batch_size", 128),
        num_workers=config.get("num_workers", 2),
        seed=config.get("seed", 19),
        calc_meanstd=config.get("calc_meanstd", True),
        normalize_mean=config.get("normalize_mean", (0.5, 0.5, 0.5)),
        normalize_std=config.get("normalize_std", (0.5, 0.5, 0.5)),
        download=False,
    )
    _, _, test_loader = data_module.build_loaders()

    model = NeuralNet(
        activation=config.get("activation", "relu"),
        use_batchnorm=config.get("use_batchnorm", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    trainer = Trainer(model=model, optimizer=None, device=device, output_dir=checkpoint_path.parent)

    print("\nRunning evaluation on CIFAR-10 test set...")
    test_metrics = trainer.evaluate_test(test_loader)

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Saved Checkpoint Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Recorded Best Validation Accuracy: {checkpoint.get('best_validation_accuracy', 0.0) * 100:.2f}%")
    print(f"Test Loss: {test_metrics['test/loss']:.4f}")
    print(f"Test Top-1 Accuracy: {test_metrics['test/accuracy'] * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()