"""
Model classes and utilities.
"""

import lightning as L
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchmetrics import Metric, MetricCollection
from losses.loss_abstract import Loss
from utils import RankedLogger

logger = RankedLogger(__name__)


class Model(L.LightningModule):
    """
    Base class for models, fitting the PyTorch Lightning interface.
    """

    encoder: nn.Module
    loss: Loss

    @torch.compiler.disable
    def log(self, *args, **kwargs):
        super().log(*args, **kwargs)

    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        encoder: nn.Module,
        loss: Loss,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # Since net and loss are nn.Modules, it is already saved in checkpoints
        # by default.
        self.save_hyperparameters(logger=False, ignore=["encoder", "loss"])

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.encoder = encoder

        self.loss = loss

        self.metrics = {
            state: {
                "batch": self.loss.build_batch_metric_aggregators(),
                "epoch": self.loss.build_epoch_metric_aggregators(),
            }
            for state in ["train", "val", "test"]
        }

        for state, metrics in self.metrics.items():
            for type, metric_set in metrics.items():
                for metric_name, metric in metric_set.items():
                    self.add_module(f"{state}_{type}_{metric_name}", metric)

        # for averaging loss across batches

    # TODO(liamhebert): Implement model logic

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The predicted values (y_hat).
        """

        return self.encoder(**x)

    def model_step(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        metrics: dict[str, MetricCollection | Metric],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | Metric | int]]:
        """Compute the loss for a batch of data.

        Args:
            batch: A dict mapping keys to data tensors for a given batch.
                Tensors are expected to have a shape of (batch_size, ...).

        Returns:
            The loss value for that batch, using self.loss.
        """
        x, y = batch["x"], batch["y"]

        node_embeddings, graph_embeddings = self.forward(x)

        loss, metrics = self.loss(node_embeddings, graph_embeddings, y, metrics)
        return loss, metrics

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks

        self.encoder.train()

        # for metrics in self.metrics.values():
        #     for metric_set in metrics.values():
        #         for metric in metric_set.values():
        #             metric.to(self.device)

        for metric in self.metrics["val"]["epoch"].values():
            metric.reset()

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute the loss and metrics for a training batch of data.

        Args:
            batch: A dict mapping keys to data tensors for a given batch.
                Tensors are expected to have a shape of (batch_size, ...).
            batch_idx: The index of the batch.

        Returns:
            The loss value for that batch, using self.loss.
        """
        loss, metrics = self.model_step(batch, self.metrics["train"]["batch"])

        weight = metrics["weight"]
        for key, metric in metrics.items():
            self.log(
                f"train/{key}",
                metric,
                on_step=(
                    True
                    if any(
                        x in key
                        for x in ["loss", "temperature", "bias", "weight"]
                    )
                    else False
                ),
                on_epoch=True,
                prog_bar=False,
                batch_size=weight,
                sync_dist=False,
            )

        self.log("train/lr", self.scheduler.get_last_lr()[0], on_step=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        """Lightning hook that is called when a training epoch ends."""
        metrics = self.metrics["train"]
        epoch_vals = self.loss.compute_epoch_metrics(
            metrics["batch"], metrics["epoch"]
        )

        for key, val in epoch_vals.items():
            self.log(
                f"train/{key}",
                val,
                prog_bar=False,
                on_epoch=True,
                sync_dist=False,
            )

        self.encoder.eval()

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute the loss and metrics for a validation batch of data.

        Args:
            batch: A dict mapping keys to data tensors for a given batch.
                Tensors are expected to have a shape of (batch_size, ...).
            batch_idx: The index of the batch.
        """
        loss, metrics = self.model_step(batch, self.metrics["val"]["batch"])

        weight: int = metrics["weight"]
        for key, metric in metrics.items():
            self.log(
                f"val/{key}",
                metric,
                on_step=True if "loss" in key else False,
                on_epoch=True,
                prog_bar=False,
                batch_size=weight,
                sync_dist=False,
            )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        metrics = self.metrics["val"]
        epoch_vals = self.loss.compute_epoch_metrics(
            metrics["batch"], metrics["epoch"]
        )

        for key, val in epoch_vals.items():
            self.log(
                f"val/{key}",
                val,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                sync_dist=False,
            )

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute the loss and metrics for a test batch of data.

        Args:
            batch: A dict mapping keys to data tensors for a given batch.
                Tensors are expected to have a shape of (batch_size, ...).
            batch_idx: The index of the batch.

        Returns:
            The loss value for that batch, using self.loss.
        """
        loss, metrics = self.model_step(batch, self.metrics["test"]["batch"])
        batch_size: int = metrics["weight"]
        for key, metric in metrics.items():
            self.log(
                f"test/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                # sync_dist=True,
            )
        return loss

    def on_test_epoch_end(self):
        """Lightning hook that is called when a test epoch ends."""
        metrics = self.metrics["test"]
        epoch_vals = self.loss.compute_epoch_metrics(
            metrics["batch"], metrics["epoch"]
        )

        for key, val in epoch_vals.items():
            self.log(
                f"test/{key}",
                val,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                # sync_dist=True,
            )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using
        DDP.

        Args:
            stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        ...
        # if stage == "fit" and self.hparams.compile:
        #     self.encoder = torch.compile(self.encoder)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate
            schedulers to be used for training.
        """

        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
        if self.hparams.scheduler is not None:
            epochs = self.trainer.max_epochs
            assert epochs is not None
            dataset_size = 380_000
            steps_per_epoch = 2886
            total_steps = epochs * steps_per_epoch  # Hard coded, sry.
            warmup = int(0.1 * total_steps)

            self.scheduler = lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1e-4,
                        end_factor=1.0,
                        total_iters=warmup,
                    ),
                    lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=total_steps - warmup, eta_min=0.0
                    ),
                ],
                milestones=[warmup],
            )

            # self.scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    # "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr_scheduler",
                },
            }
        return {"optimizer": optimizer}
