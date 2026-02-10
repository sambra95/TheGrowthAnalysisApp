"""Shared constants for TheGrowthAnalysisApp."""

# Plate layout
ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))
ALL_WELLS = [f"{r}{c}" for r in ROWS for c in COLS]

# UI color symbols for plate status
GREEN = "🟩"  # Sample present
ORANGE = "🟧"  # Not in data file
RED = "🟥"  # Excluded by user
BLUE = "🟦"  # Blank well
GRAY = "⬜"  # Not in plate map

# Legacy model type mapping
LEGACY_MODEL_TYPE_MAP = {
    "logistic": "mech_logistic",
    "gompertz": "mech_gompertz",
    "richards": "mech_richards",
    "baranyi": "mech_baranyi",
}

# Default parameters
DEFAULT_PARAMS = {
    "time_unit": "minutes",
    "pathlength_cm_": 0.42,
    "clip_time_series": (0.0, 72.0),
    "remove_wells": False,
    "blank": True,
    "window_points": 15,
    "lag_cutoff": 0.5,
    "exp_cutoff": 0.5,
    "sg_window": 15,
    "sg_poly": 2,
    "min_data_points": 5,
    "min_signal_to_noise": 1.0,
    "min_od_increase": 0.05,
    "min_growth_rate": 0.001,
    "growth_method": "Sliding Window",
    "model_family": "phenomenological",
    "model_type": "phenom_logistic",
    "phase_boundary_method": "tangent",
    "spline_s": 1.0,
}
