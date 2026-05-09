"""
RAN KPI Congestion Dashboard — entry point.

Run with:  python dashboardV3.py

The dashboard is split across these modules:
  config.py             — constants & UI styles
  geo_utils.py          — geometry helpers
  data_utils.py         — pure data utilities
  precompute.py         — data loading + precomputation
  map_builder.py        — Plotly map trace builders
  ui_components.py      — HTML card & side panel builders
  layout.py             — full Dash layout definition
  callbacks.py          — existing @callback-decorated functions
  forecast_tab_layout.py — 🔮 AI Forecasting tab layout
  forecast_callbacks.py  — 🔮 AI Forecasting tab callbacks
"""

from dash import Dash

app = Dash(
    __name__,
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"
    ],
)
app.title = "RAN KPI Dashboard"

from layout import build_layout        # noqa: E402
__import__("callbacks")                # registers existing callbacks
__import__("forecast_callbacks")       # registers forecast tab callbacks  ← NEW

app.layout = build_layout()

if __name__ == "__main__":
    app.run(debug=True)
