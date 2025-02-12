"""Module containing the VerboseTransformationPipeline class."""
from epochlib.transformation.transformation import TransformationPipeline

from src.modules.logging.logger import Logger


class VerboseTransformationPipeline(TransformationPipeline, Logger):
    """A verbose transformation pipeline that logs to the terminal and to W&B."""
