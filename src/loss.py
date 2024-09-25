"""Loss classes."""

import abc

import torch


class Loss(abc.ABC):
    """Abstract class for loss functions."""

    @abc.abstractmethod
    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss function.

        Args:
            y_pred: The predicted values.
            y_true: The true values.

        Returns:
            The loss value.
        """
        pass


# TODO(liamhebert): create whatever loss set up we want here.
class CrossEntropyLoss(Loss):
    """Cross-entropy loss function."""

    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cross-entropy loss.

        Args:
            y_pred: The predicted values.
            y_true: The true values.

        Returns:
            The cross-entropy loss value.
        """
        return torch.nn.functional.cross_entropy(y_pred, y_true)
