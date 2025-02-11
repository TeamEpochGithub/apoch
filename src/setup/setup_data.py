"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
from pathlib import Path
from typing import Any 
import pandas as pd

from src.utils.logger import logger



def setup_train_x_data(path: Path) -> Any:  # noqa: ANN401
    """Create train x data for pipeline.

    :param path: Usually raw path is a parameter
    :return: x data
    """
    logger.info("This method [setup_train_x_data] has not been changed yet, remove log statement when implemented")

    train_x_data = pd.read_csv(path)
    train_x_data = train_x_data.drop(columns=["species"]).to_numpy()

    return train_x_data


def setup_train_y_data(path: Path) -> Any:  # noqa: ANN401
    """Create train y data for pipeline.

    :param path: Usually raw path is a parameter
    :return: y data
    """
    logger.info("This method [setup_train_y_data] has not been changed yet, remove log statement when implemented")

    train_y_data = pd.read_csv(path)
    train_y_data = train_y_data["species"]
    train_y_data = pd.get_dummies(train_y_data).astype(int).to_numpy()

    return train_y_data


def setup_inference_data() -> Any:  # noqa: ANN401
    """Create data for inference with pipeline.

    :param path: Usually raw path is a parameter
    :return: Inference data
    """
    raise NotImplementedError("Setup inference data is competition specific, implement within competition repository, it might be the same as setup_train_x")


def setup_splitter_data() -> Any:  # noqa: ANN401
    """Create data for splitter.

    :return: Splitter data
    """
    raise NotImplementedError("Setup splitter data is competition specific, implement within competition repository")
