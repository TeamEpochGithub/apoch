"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
from typing import Any

import numpy as np


def setup_train_x_data(path: str) -> Any:  # noqa: ANN401
    """Create train x data for pipeline.

    :param path: Usually raw path is a parameter
    :return: x data
    """
    return np.load(path)[:, :-1]


def setup_train_y_data(path: str) -> Any:  # noqa: ANN401
    """Create train y data for pipeline.

    :param path: Usually raw path is a parameter
    :return: y data
    """
    # Load the numpy file from the path
    return np.load(path)[:, -1].astype(int)


def setup_inference_data(path: str) -> Any:  # noqa: ANN401
    """Create data for inference with pipeline.

    :param path: Usually raw path is a parameter
    :return: Inference data
    """
    return setup_train_x_data(path)


def setup_splitter_data() -> Any:  # noqa: ANN401
    """Create data for splitter.

    :return: Splitter data
    """
    return None
