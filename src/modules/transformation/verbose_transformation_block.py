"""A verbose transformation block that logs to the terminal and to W&B."""

from epochlib.transformation.transformation_block import TransformationBlock

from src.modules.logging.logger import Logger


class VerboseTransformationBlock(TransformationBlock, Logger):
    """A verbose transformation block that logs to the terminal and to W&B.

    To use this block, inherit and implement the following methods:
    - custom_transform(x: Any, **kwargs: Any) -> Any
    """
