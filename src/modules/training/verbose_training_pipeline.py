"""A verbose training pipeline that logs to the terminal and to W&B."""
from epochalyst.training import TrainingPipeline

from src.modules.logging.logger import Logger


class VerboseTrainingPipeline(TrainingPipeline, Logger):
    """A verbose training pipeline that logs to the terminal and to W&B."""
