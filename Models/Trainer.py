from pathlib import Path
import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, optimizer, device, output_dir, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss()
        self.best_validation_accuracy = -1.0
        self.best_checkpoint = self.output_dir / "best_model.pt"
        self.global_step = 0
        self.history = []

    def _run_epoch(self, loader, training):
        self.model.train(training)
        total_loss, correct, total = 0.0, 0, 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1
                total_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    def train(self, train_loader, validation_loader, epochs, config):
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = self._run_epoch(train_loader, training=True)
            val_loss, val_accuracy = self._run_epoch(validation_loader, training=False)
            row = {
                "epoch": epoch, "global_step": self.global_step,
                "train/loss": train_loss, "train/accuracy": train_accuracy,
                "validation/loss": val_loss, "validation/accuracy": val_accuracy,
            }
            self.history.append(row)
            if self.logger:
                self.logger.log_epoch(row)
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {train_loss:.4f}, train acc {train_accuracy:.4f} | "
                f"val loss {val_loss:.4f}, val acc {val_accuracy:.4f}"
            )

            if val_accuracy > self.best_validation_accuracy:
                self.best_validation_accuracy = val_accuracy
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_validation_accuracy": val_accuracy,
                }, self.best_checkpoint)
        return self.history


    def evaluate_test(self, test_loader):
        checkpoint = torch.load(self.best_checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_accuracy = self._run_epoch(test_loader, training=False)
        return {"test/loss": test_loss, "test/accuracy": test_accuracy}