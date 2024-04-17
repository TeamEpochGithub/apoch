"""Module containing functions related to creating the eda layout."""
from typing import Any

from dash import Dash

from dashboard.eda.dummy_time_series_visualizer import create_dummy_visualizer


def create_eda_layout(app: Dash) -> tuple[Any, Any]:
    """Create an eda layout of the eda tab for the dashboard.

    :param app: Dashboard app
    :return: Updated app and layout
    """
    return create_dummy_visualizer(app, file_path="./data/raw/train_eegs/*")
