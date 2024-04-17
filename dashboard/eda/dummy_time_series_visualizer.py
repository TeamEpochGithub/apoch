"""Module containing the functions to visualize parquet files."""
import glob
from typing import Any

import hydra
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from omegaconf import OmegaConf


# Define function to visualize Parquet files
def create_dummy_visualizer(app: Dash, file_path: str) -> tuple[Any, Any]:
    """Create a parquet visualizer element in the dashboard.

    :param app: The dashboard app
    :param file_path: The path to read the parquet files
    :return: Updated app and layout
    """
    # Define layout
    layout = html.Div(
        [
            html.H1("Time series Visualizer"),
            html.P("Sample ID"),
            dcc.Dropdown(
                id="Sample-dropdown",
                options=[],
                value=None,
            ),
            html.P("Feature"),
            dcc.Dropdown(
                id="feature-selector",
                options=["feature"],
                value="feature",
            ),
            dcc.Graph(id="parquet-graph"),
            dcc.Graph(id="labels-graph"),
        ],
    )

    # Register callbacks
    @app.callback(
        Output("Sample-dropdown", "options"),
        [Input("Sample-dropdown", "value")],
    )
    def update_dropdown(_: str) -> list[dict[str, str]]:
        parquet_files = glob.glob(file_path)
        return [{"label": file, "value": file} for file in parquet_files]

    @app.callback(
        Output("parquet-graph", "figure"),
        [Input("Sample-dropdown", "value"), Input("feature-selector", "value")],
    )
    def update_graph(selected_file: str, column: str) -> dict[str, Any]:
        return _update_graph(selected_file, column)


    return app, layout


def _update_graph(selected_file: str, column: str) -> dict[str, Any]:
    """Update the graph based on the selected file and column.

    :param selected_file: The selected file
    :param column: The selected column
    :return: The updated graph
    """
    if selected_file is None:
        return {}


    # Create a Plotly figure
    fig = go.Figure()
    # load the hydra yaml file dash.yaml
    cfg = OmegaConf.load("./dashboard/dash.yaml")

    # instantiate the pipeline
    pipeline = hydra.utils.instantiate(cfg.pipeline)


    # Add traces to the figure
    _add_traces(None, column, fig)


    fig.update_layout(
        title="Visualization of Dummy data",
        xaxis={"title": "Step"},
        yaxis={"title": "Amplitude"},
        showlegend=True,
    )

    return fig


def _add_traces(eeg_df: pd.DataFrame | None, column: str, fig: go.Figure) -> None:
    """Add traces to the figure.

    :param eeg_df: The EEG data
    :param column: The column to plot
    :param fig: The figure to add the traces to
    """
    eeg_df = pd.DataFrame(np.random.rand(1,1000), columns='Feature')  # noqa: NPY002
    print(eeg_df.head())
    if column in eeg_df.columns:
        scatter = go.Scatter(x=eeg_df.index, y=eeg_df[column], mode="lines", name=column)
        fig.add_trace(scatter)
        fig.update_layout(height=400)
        return

    fig.update_layout(height=2000)


# def _update_labels_graph(selected_file: str) -> dict[str, Any]:
#     """Update the labels graph based on the selected file.

#     :param selected_file: The selected file
#     :return: The updated labels graph
#     """
#     if selected_file is None:
#         return {}

#     train_df = pd.read_csv("./data/raw/train.csv")

#     # Split selected_file on '/'
#     file_name = selected_file.split("/")[-1]
#     # Assuming the eeg_id is the file name without the last 8 characters (e.g., extension)
#     eeg_id = file_name[:-8]

#     # Ensure eeg_id is compared as the correct type; casting to int might be necessary
#     # Adjust this part according to your 'eeg_id' column data type
#     try:
#         matching_eeg_id = int(eeg_id)
#     except ValueError:
#         # Handle the case where eeg_id cannot be converted to int
#         # ("eeg_id cannot be converted to an integer.")
#         return {}

#     matching = train_df[train_df["eeg_id"] == matching_eeg_id]

#     if matching.empty:
#         # (f"No matching records found for eeg_id: {eeg_id}")
#         return {}

#     columns = list(filter(lambda col: col.endswith("_vote"), matching.columns))
#     trace = go.Figure()

#     for _, x in matching.iterrows():
#         # Print statement removed to clean up output, uncomment if needed for debugging
#         # print(x[columns].to_numpy())
#         bar = go.Bar(
#             x=columns,
#             y=x[columns].to_numpy(),
#             name=f"Offset: {200 * (x['eeg_label_offset_seconds'] + 25)}",  # Optional: name each bar for clarity
#         )
#         trace.add_trace(bar)

#     layout = {
#         "title": f"Labels of {eeg_id}",
#     }

#     # Adjusted the return to provide a figure directly, which is more common with Plotly usage
#     trace.update_layout(layout)
#     return trace
