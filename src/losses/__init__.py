"""Importing all the loss functions from the losses module."""

from losses.loss_cross import NodeCrossEntropyLoss
from losses.loss_contrastive import (
    ContrastiveLossWithMetrics as ContrastiveLoss,
)
