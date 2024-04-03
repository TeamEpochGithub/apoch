"""Module for example training block."""

import dask.array as da

from src.modules.training.verbose_training_block import VerboseTrainingBlock


class ExampleTrainingBlock(VerboseTrainingBlock):
    """An example training block."""

    def train(self, x: da.Array, y: da.Array) -> tuple[da.Array, da.Array]:
        """Train the model.

        :param x: The input data
        :param y: The target data
        :return: The predictions and the target data
        """
        # ("Training")
        return x, y

    def predict(self, x: da.Array) -> da.Array:
        """Predict using the model.

        :param x: The input data
        :return: The predictions
        """
        # ("Predicting")
        return x
