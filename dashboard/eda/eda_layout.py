"""Module containing functions related to creating the eda layout."""
from typing import Any

from dash import Dash

from dashboard.eda.parquet_visualizer import create_parquet_visualizer


def create_eda_layout(app: Dash) -> tuple[Any, Any]:
    """Create an eda layout of the eda tab for the dashboard.

    :param app: Dashboard app
    :return: Updated app and layout
    """
    return create_parquet_visualizer(app, file_path="./data/raw/train_eegs/*")
