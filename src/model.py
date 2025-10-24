"""
Model classes and utilities.
"""

import lightning as L
import torch
from torch import nn
from torch import optim
from torchmetrics import Metric, MetricCollection
from losses.loss_abstract import Loss
from utils import RankedLogger
from lr_schedules.schedules import LearningRateSchedulePrototype

# from lightning.pytorch.utilities import grad_norm

logger = RankedLogger(__name__)


class Model(L.LightningModule):
    """
    Base class for models, fitting the PyTorch Lightning interface.
    """

    encoder: nn.Module
    loss: Loss
    aux_loss_weight: float = 1.0

    @torch.compiler.disable
    def log(self, *args, **kwargs):
        super().log(*args, **kwargs)

    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler: LearningRateSchedulePrototype,
        encoder: nn.Module,
        loss: Loss,
        aux_loss_weight: float = 1.0,
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
        self.aux_loss_weight = aux_loss_weight

        self.metrics = {
            state: self.loss.build_batch_metric_aggregators()
            for state in ["train", "val", "test"]
        }

        for state, metrics in self.metrics.items():
            for metric_name, metric in metrics.items():
                self.add_module(f"{state}_{metric_name}", metric)

    def replace_loss(self, loss: Loss) -> None:
        """Replace the loss function of the model. This is useful for converting
        the model to a different task.

        Args:
            loss: The new loss function to be used.
        """
        logger.warning(f"Replacing loss {type(self.loss)} with {type(loss)}")
        self.loss = loss
        for state, metrics in self.metrics.items():
            for metric_name, metric in metrics.items():
                self.__delattr__(f"{state}_{metric_name}")

        self.metrics = {
            state: self.loss.build_batch_metric_aggregators()
            for state in ["train", "val", "test"]
        }

        for state, metrics in self.metrics.items():
            for metric_name, metric in metrics.items():
                self.add_module(f"{state}_{metric_name}", metric)
        # for averaging loss across batches

    # TODO(liamhebert): Implement model logic

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        """Compute the forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The predicted values (y_hat).
        """
        # logger.warning("MODEL FORWARD")
        # print(
        #     {
        #         k: v.shape if isinstance(v, torch.Tensor) else v
        #         for k, v in x.items()
        #     },
        #     flush=True,
        # )
        # for k, v in x.items():
        #     if isinstance(v, dict):
        #         for k2, v2 in v.items():
        #             print(
        #                 f"  {k2}:"
        #                 f" {v2.shape if isinstance(v2, torch.Tensor) else v2}",
        #                 flush=True,
        #             )
        # logger.warning("okay actually starting now")
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
        self.trainer.strategy.barrier("start model step")
        x, y = batch["x"], batch["y"]

        node_embeddings, graph_embeddings, aux_loss = self.forward(x)

        loss, metrics = self.loss(
            node_embeddings, graph_embeddings, y, metrics, aux_loss
        )
        return loss, metrics

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks

        self.encoder.train()

        for metric in self.metrics["val"].values():
            metric.reset()

    def on_train_epoch_end(self):
        """Lightning hook that is called at the end of each training epoch."""
        # Reset the metrics for the next epoch
        for metric in self.metrics["train"].values():
            if isinstance(metric, MetricCollection):
                metric.reset()

    def on_validation_epoch_end(self):
        """Lightning hook that is called at the end of each training epoch."""
        # Reset the metrics for the next epoch
        for metric in self.metrics["val"].values():
            if isinstance(metric, MetricCollection):
                metric.reset()

    def on_test_epoch_end(self):
        """Lightning hook that is called at the end of each training epoch."""
        # Reset the metrics for the next epoch
        for metric in self.metrics["test"].values():
            if isinstance(metric, MetricCollection):
                metric.reset()

    def log_metrics(
        self, ret_metrics: dict[str, torch.Tensor], stage: str = "train"
    ) -> None:
        """Log the metrics for a given batch.

        Args:
            ret_metrics: A dictionary of metrics to log.
        """
        weight = ret_metrics["weight"]
        # On step metrics
        for key, metric in ret_metrics.items():
            self.log(
                f"{stage}/{key}",
                metric,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
                sync_dist=True,
            )

        # Epoch metrics
        for key, metric_set in self.metrics[stage].items():
            if isinstance(metric_set, MetricCollection):
                metrics = metric_set.compute()
                for m_key, metric in metrics.items():
                    if metric.numel() > 1:
                        # Unpack class-wise metrics into separate keys
                        assert "none" in m_key, f"Unexpected key: {m_key}"
                        for i, v in enumerate(metric):
                            class_key = m_key.replace("none", f"class_{i}")
                            self.log(
                                f"{stage}/{class_key}",
                                v,
                                on_step=False,
                                on_epoch=True,
                                prog_bar=False,
                                batch_size=weight,
                                sync_dist=True,
                            )
                    else:
                        assert (
                            metric.shape == ()
                        ), f"Unexpected shape: {metric.shape}"
                        self.log(
                            f"{stage}/{m_key}",
                            metric,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            batch_size=weight,
                            sync_dist=True,
                        )
            else:
                self.log(
                    f"{stage}/{key}",
                    metric_set,
                    on_step=False if key != "loss" else True,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=weight,
                    sync_dist=True,
                )

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
        loss, ret_metrics = self.model_step(batch, self.metrics["train"])

        self.log_metrics(ret_metrics, stage="train")

        if self.scheduler is not None:
            self.log("train/lr", self.scheduler.get_last_lr(), on_step=True)

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
        loss, ret_metrics = self.model_step(batch, self.metrics["val"])

        self.log_metrics(ret_metrics, stage="val")
        return loss

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
        loss, ret_metrics = self.model_step(batch, self.metrics["test"])

        self.log_metrics(ret_metrics, stage="test")
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
        ...
        # if stage == "fit" and self.hparams.compile:
        #     self.encoder = torch.compile(self.encoder)

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.encoder, norm_type=2)
    #     self.log_dict(norms)
    #     norms = grad_norm(self.loss, norm_type=2)
    #     self.log_dict(norms)

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
        if self.scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": self.scheduler.create_schedule(optimizer),
            }
        return {"optimizer": optimizer}
