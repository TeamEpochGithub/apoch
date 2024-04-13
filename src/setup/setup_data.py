"""This file will contain all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
from typing import Any


def setup_train_x_data() -> Any:
    """Setup data for pipeline

    :param path: Usually raw path is a parameter
    :return: Tuple containing x and y data is usually what you want to return
    """
    raise NotImplementedError("Setup train data x is competition specific, implement within competition repository")


def setup_train_y_data() -> Any:
    raise NotImplementedError("Setup train data y is competition specific, implement within competition repository")


def setup_inference_data() -> Any:
    raise NotImplementedError("Setup inference data is competition specific, implement within competition repository, it might be the same as setup_train_x")


def setup_splitter_data() -> Any:
    raise NotImplementedError("Setup splitter data is competition specific, implement within competition repository")
