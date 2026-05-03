"""
Geometry helpers.
"""

import pandas as pd

from config import ALLOWED_LEVELS


def normalize_level(level) -> str:
    if pd.isna(level):
        return "Not Congested"
    raw   = str(level).strip()
    upper = raw.upper()
    if upper == "P0": return "P0"
    if upper == "P1": return "P1"
    title = raw.title()
    if title in {"High", "Critical", "Severe"}: return "P0"
    if title in {"Medium"}:                     return "P1"
    if title in {"Low", "Not Congested", "Unknown"}: return "Not Congested"
    if raw in ALLOWED_LEVELS:                   return raw
    return "Not Congested"
