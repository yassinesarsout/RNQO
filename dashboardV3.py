"""
RAN KPI Congestion Dashboard — entry point.

Run with:  python dashboardV3.py

The dashboard is split across these modules:
  config.py        — constants & UI styles
  geo_utils.py     — geometry helpers (vectorized sector edges, zoom, etc.)
  data_utils.py    — pure data utilities (load, filter, sample, empty figures)
  precompute.py    — data loading + module-level precomputation (polygons, caches)
  map_builder.py   — Plotly map trace builders + figure assembly
  ui_components.py — HTML card & side panel builders
  layout.py        — full Dash layout definition
  callbacks.py     — all @callback-decorated functions
"""

from dash import Dash

app = Dash(__name__)
app.title = "RAN KPI Dashboard"

# Import layout and callbacks AFTER the app is created so that
# @callback decorators attach to the correct app context.
from layout import build_layout  # noqa: E402
__import__("callbacks")  # side-effect: registers all @callback decorators

app.layout = build_layout()

if __name__ == "__main__":
    app.run(debug=True)
