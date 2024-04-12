"""Module for example training block."""
from dataclasses import dataclass

from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from src.modules.logging.logger import Logger
@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block for training EEG / Spectrogram models.
    """


