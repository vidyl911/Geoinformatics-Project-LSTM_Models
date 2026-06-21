# ============================================================
# config.py — Shared configuration for NDVI/NDMI gap-fill pipeline
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ─────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────
DATA_PATH = r"F:/Geoinformatics ProjectWork/outputs/cleaned_full_dataset.csv"
OUT_DIR   = r"F:/Geoinformatics ProjectWork/outputs/lstm_results_advanced"
MODEL_DIR = os.path.join(OUT_DIR, "models")
PLOT_DIR  = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Model hyper-parameters ───────────────────────────────────
TIMESTEPS = 7
HORIZON   = 1
BATCH     = 32
EPOCHS    = 50        # lower to 3 for quick smoke-test
VAL_SPLIT = 0.15
SEED      = 42

# ── Feature columns (will be filtered to those present in CSV) ──
ALL_FEATURE_COLS = [
    "NDVI", "NDMI",
    "Temperatura", "Precipitazione", "Umidita_Relativa",
    "Elevazione", "Slope", "Northerness", "Northerness_Slope",
    "sin_DoY", "cos_DoY",
    "NDVI_lag1", "NDVI_lag7",
    "NDMI_lag1", "NDMI_lag7",
    "Temp_lag1", "Precip_sum7",
    "NDVI_roll7", "NDMI_roll7",
]

TARGET_NDVI = "NDVI"
TARGET_NDMI = "NDMI"

# ── Premium colour palette ───────────────────────────────────
PALETTE = {
    "lstm"     : "#1A6B9A",
    "bilstm"   : "#D4550B",
    "smoothed" : "#2E8B57",
    "attn"     : "#7B2D8B",
    "observed" : "#2C3E50",
    "gap"      : "#E74C3C",
    "fill"     : "#27AE60",
    "train"    : "#3498DB",
    "val"      : "#E67E22",
    "bg"       : "#F8F9FA",
    "grid"     : "#DEE2E6",
    "text"     : "#212529",
    "accent"   : "#FFC107",
}

SEASON_COLOURS = {
    "Winter": "#5B9BD5",
    "Spring": "#70AD47",
    "Summer": "#ED7D31",
    "Autumn": "#A5A5A5",
}

MODEL_COLOURS = {
    "LSTM"             : PALETTE["lstm"],
    "BiLSTM"           : PALETTE["bilstm"],
    "Smoothed-BiLSTM"  : PALETTE["smoothed"],
    "Attention-BiLSTM" : PALETTE["attn"],
}

MODEL_MARKERS = {
    "LSTM": "o", "BiLSTM": "s",
    "Smoothed-BiLSTM": "^", "Attention-BiLSTM": "D",
}

MODEL_LS = {
    "LSTM": "-", "BiLSTM": "--",
    "Smoothed-BiLSTM": "-.", "Attention-BiLSTM": ":",
}

SEASON_MAP = [
    (range(1, 60),   "Winter"),
    (range(60, 152), "Spring"),
    (range(152,244), "Summer"),
    (range(244,335), "Autumn"),
    (range(335,366), "Winter"),
]

# ── Global plot style ────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : PALETTE["bg"],
    "axes.facecolor"    : "white",
    "axes.edgecolor"    : PALETTE["grid"],
    "axes.labelcolor"   : PALETTE["text"],
    "axes.labelsize"    : 11,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.grid"         : True,
    "grid.color"        : PALETTE["grid"],
    "grid.linewidth"    : 0.6,
    "grid.alpha"        : 0.7,
    "xtick.color"       : PALETTE["text"],
    "ytick.color"       : PALETTE["text"],
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.framealpha" : 0.92,
    "legend.edgecolor"  : PALETTE["grid"],
    "font.family"       : "DejaVu Sans",
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "savefig.facecolor" : PALETTE["bg"],
})
