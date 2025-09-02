"""
Simple fully-connected neural net for computing predictions.
"""

import torch
from torch import nn


class SimpleOutputHead(nn.Module):
    """
    A simple fully-connected neural net for computing predictions.
    """

    model: nn.Module
    input_dim: int
    output_dim: int

    def __init__(
        self,
        input_dim: int = 784,
        output_dim: int = 10,
    ) -> None:
        """Initialize a `SimpleOutputHead` module.

        Args:
            input_dim: The number of input features.
            output_dim: The number of output features of the linear layer.
        """
        super().__init__()

        self.model = nn.Linear(input_dim, output_dim)
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
