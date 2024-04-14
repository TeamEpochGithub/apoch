"""MSE scorer."""

from typing import Any

import numpy as np

from src.scoring.scorer import Scorer


class Accuracy(Scorer):
    """Abstract scorer class from which other scorers inherit from."""

    def __init__(self, name: str) -> None:
        """Initialize the scorer with a name."""
        self.name = name

    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], **kwargs: Any) -> float:
        """Calculate the score."""
        # Apply a threshold of 0.5 to the predictions
        # Sqeeze the predictions to a 1D array
        y_pred = np.squeeze(y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        # Calculate the accuracy
        return np.mean(y_true == y_pred)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name
