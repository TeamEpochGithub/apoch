"""Small torch network for testing purposes."""
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class MLP(nn.Module):
    """Simple MLP model."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the model."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # apply softmax
        return F.sigmoid(x)