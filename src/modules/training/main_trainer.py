"""Module for example training block."""
from dataclasses import dataclass

from epochalyst.pipeline.model.training.torch_trainer import T, T_co, TorchTrainer, TrainTestDataset
from torch.utils.data import Dataset

from src.modules.logging.logger import Logger


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Dummy Trainer model."""
