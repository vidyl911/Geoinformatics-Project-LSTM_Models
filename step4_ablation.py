#!/usr/bin/env python3
"""
step4_ablation.py
=================
NDVI / NDMI Gap-Fill Pipeline — Step 4
  • Timestep ablation study
      Incrementally zero-pad leading timesteps and record RMSE
      to show how many past time-steps each model relies on.
      Inspired by Farbo et al. (2024) Fig. 12 — extended.

Run:
    python step4_ablation.py

Outputs:
    OUT_DIR/ablation_timesteps.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import tensorflow as tf

from config import (
    OUT_DIR, MODEL_DIR, SEED,
    TARGET_NDVI, TIMESTEPS, HORIZON,
)

tf.random.set_seed(SEED)
np.random.seed(SEED)

ARR_DIR = os.path.join(OUT_DIR, "arrays")


# ─────────────────────────────────────────────────────────────
# 1. LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────
print("\n📂  Loading artefacts …")

with open(os.path.join(ARR_DIR, "feature_cols.txt")) as fh:
    FEATURE_COLS = [l.strip() for l in fh if l.strip()]

scaler_NDVI = joblib.load(os.path.join(MODEL_DIR, "scaler_NDVI.pkl"))

test_df = pd.read_csv(
    os.path.join(OUT_DIR, "test_df.csv"), parse_dates=["Date"]
)
for col in FEATURE_COLS:
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").astype(np.float32)
if "DoY" not in test_df.columns:
    test_df["DoY"] = test_df["Date"].dt.dayofyear

MODEL_FILES = {
    "LSTM":             "LSTM_NDVI.keras",
    "BiLSTM":           "BiLSTM_NDVI.keras",
    "Smoothed-BiLSTM":  "SBiLSTM_NDVI.keras",
    "Attention-BiLSTM": "ABiLSTM_NDVI.keras",
}

models = {}
for lbl, fname in MODEL_FILES.items():
    models[lbl] = tf.keras.models.load_model(os.path.join(MODEL_DIR, fname))
    print(f"✔  {lbl} loaded")


# ─────────────────────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────────────────────
def build_sequences(arr: np.ndarray, target_idx: int):
    X, y = [], []
    for i in range(TIMESTEPS, len(arr) - HORIZON + 1):
        X.append(arr[i - TIMESTEPS : i, :])
        y.append(arr[i + HORIZON - 1, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def inv_scale(y_sc, scaler, feat_cols, target):
    dummy = np.zeros((len(y_sc), len(feat_cols)), dtype=np.float32)
    tidx  = feat_cols.index(target)
    dummy[:, tidx] = y_sc.ravel()
    return scaler.inverse_transform(dummy)[:, tidx]


def collect_test_sequences(src_df, scaler, feat_cols, target):
    """Combine sequences across all PIDs into a single array pair."""
    all_X, all_y = [], []
    for pid, pdf in src_df.groupby("PID"):
        pdf = pdf.sort_values("Date").copy()
        for col in feat_cols:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
        pdf = pdf.dropna(subset=feat_cols)
        if len(pdf) < TIMESTEPS + 1:
            continue
        arr  = scaler.transform(pdf[feat_cols].values.astype(np.float32))
        tidx = feat_cols.index(target)
        X, y = build_sequences(arr, tidx)
        all_X.append(X)
        all_y.append(y)
    if not all_X:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(all_X), np.concatenate(all_y)


# ─────────────────────────────────────────────────────────────
# 3. ABLATION STUDY
# ─────────────────────────────────────────────────────────────
print("\n🔬  Building test sequences for ablation …")
X_all, y_all = collect_test_sequences(test_df, scaler_NDVI, FEATURE_COLS, TARGET_NDVI)
y_inv = inv_scale(y_all, scaler_NDVI, FEATURE_COLS, TARGET_NDVI)
print(f"✔  Sequences: {X_all.shape}")


def timestep_ablation(model, model_label: str, max_ablate: int = TIMESTEPS):
    """
    Progressively zero-pad the first n_ablate timesteps of each sequence
    and record RMSE. n_ablate=0 means no ablation (baseline).
    """
    results = []
    for n_ablate in range(0, max_ablate + 1):
        X_pert = X_all.copy()
        if n_ablate > 0:
            X_pert[:, :n_ablate, :] = 0.0
        yp_sc  = model.predict(X_pert, verbose=0).ravel()
        yp_inv = inv_scale(yp_sc, scaler_NDVI, FEATURE_COLS, TARGET_NDVI)
        rmse   = np.sqrt(mean_squared_error(y_inv, yp_inv))
        results.append({
            "n_ablated": n_ablate,
            "RMSE":      rmse,
            "Model":     model_label,
        })
        print(f"   {model_label}  n_ablated={n_ablate}  RMSE={rmse:.5f}")
    return pd.DataFrame(results)


print("\n🔬  Running timestep ablation …")
ablation_frames = []
for lbl, model in models.items():
    print(f"\n  ── {lbl}")
    abl = timestep_ablation(model, lbl)
    ablation_frames.append(abl)

ablation_df = pd.concat(ablation_frames, ignore_index=True)
out_path    = os.path.join(OUT_DIR, "ablation_timesteps.csv")
ablation_df.to_csv(out_path, index=False)
print(f"\n✔  Ablation results saved → {out_path}")

print("\n" + "="*60)
print("  ✅  STEP 4 COMPLETE — run step5_plots.py next")
print("="*60)
