"""Importing all the loss functions from the losses module."""

from losses.loss_cross import NodeCrossEntropyLoss
from losses.loss_contrastive import (
    ContrastiveLossWithMetrics as ContrastiveLoss,
)
from losses.loss_sup_contrastive import (
    SupContrastiveLossWithMetrics as SupContrastiveLoss,
)

from losses.loss_anchor_contrastive import (
    AnchorContrastiveLossWithMetrics as AnchorContrastiveLoss,
)

# Combined supervised CE + SupCon loss
from losses.loss_supcon_ce import SupConWithCELoss
