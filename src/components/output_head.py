"""Simple fully-connected neural net for computing predictions."""

import torch
from torch import nn


class SimpleOutputHead(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    model: nn.Module
    input_dim: int
    output_dim: int

    def __init__(
        self,
        input_dim: int = 784,
        lin1_dim: int = 256,
        lin2_dim: int = 256,
        lin3_dim: int = 256,
        output_dim: int = 10,
    ) -> None:
        """Initialize a `SimpleOutputHead` module.

        Args:
            input_dim: The number of input features.
            lin1_dim: The number of output features of the first linear layer.
            lin2_dim: The number of output features of the second linear layer.
            lin3_dim: The number of output features of the third linear layer.
            output_dim: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, lin1_dim),
            nn.BatchNorm1d(lin1_dim),
            nn.ReLU(),
            nn.Linear(lin1_dim, lin2_dim),
            nn.BatchNorm1d(lin2_dim),
            nn.ReLU(),
            nn.Linear(lin2_dim, lin3_dim),
            nn.BatchNorm1d(lin3_dim),
            nn.ReLU(),
            nn.Linear(lin3_dim, output_dim),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x: The input tensor.

        Returns:
            A tensor of predictions.
        """
        assert (
            x.shape[1] == self.input_dim
        ), f"Expected input shape {self.input_dim}, got {x.shape[1]}"

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleOutputHead()
