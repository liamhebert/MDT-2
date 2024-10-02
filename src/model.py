"""Model classes and utilities."""

from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from components.output_head import SimpleOutputHead
from losses.loss_abstract import Loss


class Model(pl.LightningModule):
    """Base class for models, fitting the PyTorch Lightning interface."""

    net: nn.Module
    loss: Loss

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        net: nn.Module,
        loss: Loss,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # Since net and loss are nn.Modules, it is already saved in checkpoints
        # by default.
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.loss = loss

        self.metrics = {
            state: {
                "batch": self.loss.build_batch_metric_aggregators(),
                "epoch": self.loss.build_epoch_metric_aggregators(),
            }
            for state in ["train", "val", "test"]
        }

        # for averaging loss across batches

    # TODO(liamhebert): Implement model logic

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The predicted values (y_hat).
        """
        # TODO(liamhebert): Implement forward pass with updated args
        return self.net(x), self.net(x)

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
        node_embeddings, graph_embeddings = self.forward(x)
        loss = self.loss(
            node_embeddings,
            graph_embeddings,
            y,
        )
        return loss

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
    ) -> torch.Tensor:
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
        return loss

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
        return loss

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
