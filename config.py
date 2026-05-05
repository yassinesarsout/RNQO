"""
Dashboard constants — data paths, KPI defaults, congestion tiers,
cache limits, and UI styles.
"""

DATA_PATH = "datasets/kpi_with_congestion.csv"

MAX_SCATTER_POINTS = 20000
MAX_HIST_POINTS    = 50000
TABLE_PAGE_SIZE    = 20

DEFAULT_KPI               = "DL PRB Load"
DEFAULT_KPI_2             = "DL Avg Throughput"
DEFAULT_KPI_3             = "Active Users"
DEFAULT_X                 = "DL PRB Load"
DEFAULT_Y                 = "DL Avg Throughput"
DEFAULT_COLOR             = "Active Users"
DEFAULT_PRB_THRESHOLD     = 90
DEFAULT_THROUGHPUT_THRESHOLD = 4000
DEFAULT_ACTIVE_USER_THRESHOLD = 4

# ---- Congestion tiers ----
ALLOWED_LEVELS = ["Not Congested", "P1", "P0"]
LEVEL_ORDER    = {"Not Congested": 0, "P1": 1, "P0": 2}
CONGESTION_COLORS = {
    "P0":            "#ef4444",
    "P1":            "#f97316",
    "Not Congested": "#22c55e",
}
P0_COLOR = CONGESTION_COLORS["P0"]
P1_COLOR = CONGESTION_COLORS["P1"]

_FILTER_CACHE_MAX = 8

# ---- Geo / map configuration ----
GEO_DATA_PATH    = "datasets/final_info_data.csv"
SITE_KEY_SEP     = "||"
FIXED_HALF_ANGLE = 32       # degrees, ± around azimuth
CONE_POINTS      = 12       # arc sample points per sector
SECTOR_ZOOM_THRESHOLD = 13  # show sectors at this Leaflet zoom and above
NO_DATA_COLOR    = "#94a3b8"

# Base range (m) by coverage type — NOT from expected_cell_range column
COVERAGE_RANGES = {
    "Dense Urban": 350,
    "Urban":       550,
    "Suburban":    950,
    "Rural":      1700,
}
DEFAULT_CONE_RANGE   = 700   # used when coverage type is unknown
CELL_RANGE_INCREMENT = 160   # extra meters per cell index within same site

# ---- UI styles ----
CARD_STYLE = {
    "padding": "14px 16px",
    "border": "1px solid #e5e7eb",
    "borderRadius": "14px",
    "background": "white",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
}
SECTION_STYLE = {
    "padding": "14px",
    "border": "1px solid #e5e7eb",
    "borderRadius": "16px",
    "background": "white",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    "marginBottom": "16px",
}
APP_STYLE = {
    "fontFamily": "Inter, Arial, sans-serif",
    "backgroundColor": "#f8fafc",
    "minHeight": "100vh",
    "padding": "18px",
}
_PILL_BTN = {
    "padding": "5px 14px",
    "borderRadius": "20px",
    "border": "1px solid #e2e8f0",
    "fontSize": "12px",
    "cursor": "pointer",
    "background": "white",
    "fontFamily": "Inter, Arial, sans-serif",
}
_PRIMARY_BTN = {
    **_PILL_BTN,
    "background": "#0f172a",
    "color": "white",
    "border": "1px solid #0f172a",
    "fontWeight": 600,
}
