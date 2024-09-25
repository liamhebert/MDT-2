"""Model classes and utilities."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics import MaxMetric
from torchmetrics import MeanMetric

from components.simple_dense_net import SimpleDenseNet
from loss import Loss


class Model(pl.LightningModule):
    """Base class for models, fitting the PyTorch Lightning interface."""

    net: SimpleDenseNet
    loss: Loss

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        net: torch.nn.Module,
        loss: Loss,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # Since net is a nn.Module, it is already saved in checkpoints by
        # default.
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.loss = loss

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MaxMetric()

    # TODO(liamhebert): Implement model logic

    def forward(self, x) -> torch.Tensor:
        """Compute the forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The predicted values (y_hat).
        """
        # TODO(liamhebert): Implement forward pass with updated args
        return self.net(x)

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the loss for a batch of data.

        Args:
            batch: A dict mapping keys to data tensors for a given batch.
                Tensors are expected to have a shape of (batch_size, ...).

        Returns:
            The loss value for that batch, using self.loss.
        """
        x, y = batch["x"], batch["y"]
        logits = self.forward(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

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
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        """Compute the loss and metrics for a validation batch of data.

        Args:
            batch: A dict mapping keys to data tensors for a given batch.
                Tensors are expected to have a shape of (batch_size, ...).
            batch_idx: The index of the batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_loss_best` as a value through `.compute()` method, instead of
        # as a metric object otherwise metric would be reset by lightning after
        # each epoch
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
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
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
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
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
