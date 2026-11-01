"""W&B integration. The project still runs if W&B is disabled."""
import wandb


class WandbLogger:
    def __init__(self, enabled, project, config, name=None, entity=None):
        self.run = None
        if enabled:
            self.run = wandb.init(project=project, entity=entity, name=name, config=config,
                                  job_type="train", mode="online", reinit="finish_previous")
            # Metrics are logged once per epoch; global_step is the cumulative batch count.
            self.run.define_metric("global_step")
            for metric in ("train/loss", "train/accuracy", "validation/loss",
                           "validation/accuracy", "test/loss", "test/accuracy"):
                self.run.define_metric(metric, step_metric="global_step")

    def log_epoch(self, metrics):
        if self.run is not None:
            self.run.log(metrics)

    def log_summary(self, values):
        if self.run is not None:
            for key, value in values.items():
                self.run.summary[key] = value

    def log_model(self, checkpoint_path):
        if self.run is not None:
            artifact = wandb.Artifact("best-cifar10-model", type="model")
            artifact.add_file(str(checkpoint_path))
            self.run.log_artifact(artifact)

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
