"""Script to startup dash dashboard."""
from typing import Any

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from dashboard.eda.eda_layout import create_eda_layout
from dashboard.eda.dummy_time_series_visualizer import create_dummy_visualizer

# Initialize the Dash app
app = dash.Dash(__name__)

parquet_app, parquet_layout = create_dummy_visualizer(app=app, file_path="data/raw/train_eegs/*")

# Define the layout of the dashboard
app.layout = html.Div(
    [
        html.H1("Q3 - Harmful Brain Activity", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            value="eda",
            children=[
                dcc.Tab(label="EDA", value="eda"),
                dcc.Tab(label="Model", value="model"),
                dcc.Tab(label="Predictions", value="predictions"),
            ],
        ),
        html.Div(id="tabs-content"),
    ],
)


@app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
def render_content(tab: str) -> dict[str, Any]:
    """Render_content renders content.

    :param tab: Tab to render
    :return: layout
    """
    if tab == "eda":
        eda_app, eda_layout = create_eda_layout(app=app)
        return eda_layout

    return {}


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
