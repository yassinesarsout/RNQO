"""
Dashboard constants — data paths, KPI defaults, congestion tiers,
cache limits, and UI styles.
"""

DATA_PATH     = "datasets/kpi_with_congestion.csv"
RAW_DATA_PATH = "datasets/final_kpi_data.csv"     # input to congestion engine

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
SECTOR_ZOOM_THRESHOLD = 15  # show sectors at this Leaflet zoom and above
NO_DATA_COLOR    = "#94a3b8"

# ---- Cone radius — derived from Average Timing Advance (already in metres) ----
TA_COLUMN        = "Average Timing Advance"   # KPI column name
MIN_CONE_RANGE_M   =  40.0   # floor  (very small urban cells)
MAX_CONE_RANGE_M   = 250.0   # ceiling — ~52 px at zoom 15
DEFAULT_CONE_RANGE = 100.0   # fallback when TA is unavailable for a cell

# ---- Approach 2: Frequency-based cone sizing ----
# Lower frequency → longest + narrowest (far-reaching spike)
# Higher frequency → shortest + widest  (broad close-in fan)
# Keys: integer MHz value or the string "tdd"
FREQ_CONE_PARAMS: dict = {
    800:   {"range_m": 250, "half_angle": 12},
    1800:  {"range_m": 170, "half_angle": 23},
    2100:  {"range_m": 120, "half_angle": 34},
    "tdd": {"range_m": 100, "half_angle": 28},
}
FREQ_CONE_DEFAULT: dict = {"range_m": 120, "half_angle": 30}  # unknown frequency

# ── Design tokens ────────────────────────────────────────────────────────────
ACCENT       = "#6366f1"   # indigo
ACCENT_DARK  = "#4338ca"
ACCENT_LIGHT = "#eef2ff"
SURFACE      = "#ffffff"
SURFACE_2    = "#f8fafc"
BORDER_COLOR = "#e2e8f0"
TEXT_PRIMARY = "#0f172a"
TEXT_MUTED   = "#64748b"

# ── Component styles ─────────────────────────────────────────────────────────
CARD_STYLE = {
    "padding": "18px 20px",
    "border": "none",
    "borderRadius": "16px",
    "background": SURFACE,
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
}
SECTION_STYLE = {
    "padding": "20px",
    "border": "none",
    "borderRadius": "20px",
    "background": SURFACE,
    "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
    "marginBottom": "20px",
}
APP_STYLE = {
    "fontFamily": "'Inter', 'Segoe UI', system-ui, Arial, sans-serif",
    "backgroundColor": "#f1f5f9",
    "minHeight": "100vh",
}
_PILL_BTN = {
    "padding": "6px 14px",
    "borderRadius": "20px",
    "border": f"1px solid {BORDER_COLOR}",
    "fontSize": "12px",
    "cursor": "pointer",
    "background": SURFACE,
    "fontFamily": "'Inter', system-ui, sans-serif",
    "fontWeight": 500,
    "color": TEXT_PRIMARY,
}
_PRIMARY_BTN = {
    **_PILL_BTN,
    "background": TEXT_PRIMARY,
    "color": "white",
    "border": f"1px solid {TEXT_PRIMARY}",
    "fontWeight": 600,
}
_ACCENT_BTN = {
    **_PILL_BTN,
    "background": ACCENT,
    "color": "white",
    "border": f"1px solid {ACCENT}",
    "fontWeight": 600,
}

_TAB_STYLE = {
    "padding": "25px 22px",
    "fontWeight": 500, "fontSize": "13px",
    "color": TEXT_MUTED,
    "border": "none",
    "borderBottom": "2px solid transparent",
    "background": "transparent",
    "fontFamily": "'Inter', system-ui, sans-serif",
}
_TAB_SELECTED = {
    **_TAB_STYLE,
    "color": ACCENT,
    "fontWeight": 700,
    "borderBottom": f"2px solid {ACCENT}",
    "background": SURFACE,
    "borderRadius": "10px 10px 0 0",
}
