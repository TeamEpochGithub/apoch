"""Module for example training block."""
from dataclasses import dataclass

from epochalyst.pipeline.model.training.torch_trainer import T, T_co, TorchTrainer, TrainTestDataset
from torch.utils.data import Dataset

from src.modules.logging.logger import Logger


@dataclass
class DummyTrainer(TorchTrainer, Logger):
    """Dummy Trainer model."""

    def _concat_datasets(
        self,
        train_dataset: T,
        test_dataset: T,
        train_indices: list[int],
        test_indices: list[int],
    ) -> Dataset[T_co]:
        """OVerride this method because epochalsyst is broken.

        :param train_dataset: The training dataset.
        :param test_dataset: The test dataset.
        :param train_indices: The indices for the training data.
        :param test_indices: The indices for the test data.
        :return: A new dataset containing the concatenated data in the original order.
        """
        return TrainTestDataset(train_dataset, test_dataset, list(train_indices), list(test_indices))
