"""Module containing the functions to visualize parquet files."""
import glob
from typing import Any

import hydra
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from omegaconf import OmegaConf


# Define function to visualize Parquet files
def create_parquet_visualizer(app: Dash, file_path: str) -> tuple[Any, Any]:
    """Create a parquet visualizer element in the dashboard.

    :param app: The dashboard app
    :param file_path: The path to read the parquet files
    :return: Updated app and layout
    """
    # Define layout
    layout = html.Div(
        [
            html.H1("EEG Visualizer"),
            html.P("EEG id"),
            dcc.Dropdown(
                id="parquet-dropdown",
                options=[],
                value=None,
            ),
            html.P("Feature"),
            dcc.Dropdown(
                id="parquet-column-selector",
                options=["bipolar", "raw", "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"],
                value="Fp1",
            ),
            dcc.Graph(id="parquet-graph"),
            dcc.Graph(id="labels-graph"),
        ],
    )

    # Register callbacks
    @app.callback(
        Output("parquet-dropdown", "options"),
        [Input("parquet-dropdown", "value")],
    )
    def update_dropdown(_: str) -> list[dict[str, str]]:
        parquet_files = glob.glob(file_path)
        return [{"label": file, "value": file} for file in parquet_files]

    @app.callback(
        Output("parquet-graph", "figure"),
        [Input("parquet-dropdown", "value"), Input("parquet-column-selector", "value")],
    )
    def update_graph(selected_file: str, column: str) -> dict[str, Any]:
        return _update_graph(selected_file, column)

    @app.callback(
        Output("labels-graph", "figure"),
        [Input("parquet-dropdown", "value")],
    )
    def update_label_graph(selected_file: str) -> dict[str, Any]:
        return _update_labels_graph(selected_file)

    return app, layout


def _update_graph(selected_file: str, column: str) -> dict[str, Any]:
    """Update the graph based on the selected file and column.

    :param selected_file: The selected file
    :param column: The selected column
    :return: The updated graph
    """
    if selected_file is None:
        return {}

    eeg_df = pd.read_parquet(selected_file)

    # Split selected_file on '/'
    file_name = selected_file.split("/")[-1]
    # Assuming the eeg_id is the file name without the last 8 characters (e.g., extension)
    eeg_id = file_name[:-8]

    # Ensure eeg_id is compared as the correct type; casting to int might be necessary
    # Adjust this part according to your 'eeg_id' column data type
    try:
        matching_eeg_id = int(eeg_id)
    except ValueError:
        # Handle the case where eeg_id cannot be converted to int
        # ("eeg_id cannot be converted to an integer.")
        return {}

    train_df = pd.read_csv("./data/raw/train.csv")
    matching = train_df[train_df["eeg_id"] == matching_eeg_id]

    # Create a Plotly figure
    fig = go.Figure()
    # load the hydra yaml file dash.yaml
    cfg = OmegaConf.load("./dashboard/dash.yaml")

    # instantiate the pipeline
    pipeline = hydra.utils.instantiate(cfg.pipeline)

    # create an XData object with a single EEG
    X = XData(meta=pd.DataFrame(), eeg={0: eeg_df}, kaggle_spec=None, eeg_spec=None, shared=None)

    # process the EEG data
    X = pipeline.transform(X)
    if X.eeg is None:
        raise ValueError("No EEG data found.")
    eeg_df = X.eeg[0]

    # Add traces to the figure
    _add_traces(eeg_df, column, fig)

    # Add vertical lines for each eeg_label_offset_seconds
    for offset in matching["eeg_label_offset_seconds"]:
        fig.add_shape(type="line", x0=(offset + 25) * 200, y0=0, x1=(offset + 25) * 200, y1=1, xref="x", yref="paper", line={"color": "Red", "width": 2})

    fig.update_layout(
        title=f"Visualization of {eeg_id}",
        xaxis={"title": "Step (200Hz)"},
        yaxis={"title": "µV"},
        showlegend=True,
    )

    return fig


def _add_traces(eeg_df: pd.DataFrame, column: str, fig: go.Figure) -> None:
    """Add traces to the figure.

    :param eeg_df: The EEG data
    :param column: The column to plot
    :param fig: The figure to add the traces to
    """
    if column in eeg_df.columns:
        scatter = go.Scatter(x=eeg_df.index, y=eeg_df[column], mode="lines", name=column)
        fig.add_trace(scatter)
        fig.update_layout(height=400)
        return

    if column == "raw":
        eeg_df = process_raw_df(eeg_df)
    elif column == "bipolar":
        if "LT1" not in eeg_df.columns:
            eeg_df = to_bipolar(eeg_df)
        eeg_df = process_bipolar_eeg(eeg_df)
    else:
        raise ValueError(f"Column {column} not recognized.")
    for col in eeg_df.columns:
        scatter = go.Scatter(x=eeg_df.index, y=eeg_df[col], mode="lines", name=col)
        fig.add_trace(scatter)
    fig.update_layout(height=2000)


def _update_labels_graph(selected_file: str) -> dict[str, Any]:
    """Update the labels graph based on the selected file.

    :param selected_file: The selected file
    :return: The updated labels graph
    """
    if selected_file is None:
        return {}

    train_df = pd.read_csv("./data/raw/train.csv")

    # Split selected_file on '/'
    file_name = selected_file.split("/")[-1]
    # Assuming the eeg_id is the file name without the last 8 characters (e.g., extension)
    eeg_id = file_name[:-8]

    # Ensure eeg_id is compared as the correct type; casting to int might be necessary
    # Adjust this part according to your 'eeg_id' column data type
    try:
        matching_eeg_id = int(eeg_id)
    except ValueError:
        # Handle the case where eeg_id cannot be converted to int
        # ("eeg_id cannot be converted to an integer.")
        return {}

    matching = train_df[train_df["eeg_id"] == matching_eeg_id]

    if matching.empty:
        # (f"No matching records found for eeg_id: {eeg_id}")
        return {}

    columns = list(filter(lambda col: col.endswith("_vote"), matching.columns))
    trace = go.Figure()

    for _, x in matching.iterrows():
        # Print statement removed to clean up output, uncomment if needed for debugging
        # print(x[columns].to_numpy())
        bar = go.Bar(
            x=columns,
            y=x[columns].to_numpy(),
            name=f"Offset: {200 * (x['eeg_label_offset_seconds'] + 25)}",  # Optional: name each bar for clarity
        )
        trace.add_trace(bar)

    layout = {
        "title": f"Labels of {eeg_id}",
    }

    # Adjusted the return to provide a figure directly, which is more common with Plotly usage
    trace.update_layout(layout)
    return trace
