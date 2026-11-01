"""Create local plots from epoch history."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_history_plots(self, history):
        epochs = [row["epoch"] for row in history]
        for filename, train_key, val_key, ylabel, title in [
            ("loss_curves.png", "train/loss", "validation/loss", "Loss", "Training and validation loss"),
            ("accuracy_curves.png", "train/accuracy", "validation/accuracy", "Accuracy", "Training and validation accuracy"),
        ]:
            plt.figure()
            plt.plot(epochs, [row[train_key] for row in history], label="Training")
            plt.plot(epochs, [row[val_key] for row in history], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / filename, dpi=150)
            plt.close()
