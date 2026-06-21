#!/usr/bin/env python3
"""
step1_data_prep.py
==================
NDVI / NDMI Gap-Fill Pipeline — Step 1
  • Load raw CSV
  • Sanitise dtypes
  • Apply Blackman FIR smoothing  (Dey et al. 2024)
  • Build train / test splits
  • Fit & save MinMaxScalers
  • Build and persist sequence arrays (X_train, y_train, X_test, y_test)
    for both NDVI and NDMI targets

Run:
    python step1_data_prep.py

Outputs (written to OUT_DIR / MODEL_DIR):
    scalers/scaler_NDVI.pkl
    scalers/scaler_NDMI.pkl
    arrays/X_tr_ndvi.npy  y_tr_ndvi.npy
    arrays/X_te_ndvi.npy  y_te_ndvi.npy
    arrays/X_tr_ndmi.npy  y_tr_ndmi.npy
    arrays/X_te_ndmi.npy  y_te_ndmi.npy
    arrays/feature_cols.txt        (one feature name per line)
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import joblib

# ── import shared config ─────────────────────────────────────
from config import (
    DATA_PATH, OUT_DIR, MODEL_DIR, PLOT_DIR,
    ALL_FEATURE_COLS, TARGET_NDVI, TARGET_NDMI,
    TIMESTEPS, HORIZON, SEED,
)

np.random.seed(SEED)

ARR_DIR = os.path.join(OUT_DIR, "arrays")
os.makedirs(ARR_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
print(f"\n📂  Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values(["PID", "Date"]).reset_index(drop=True)
df["DoY"]  = df["Date"].dt.dayofyear
df["Year"] = df["Date"].dt.year

# Nuclear dtype sanitisation — everything → float32 except Date / PID
_skip = {"Date", "PID"}
for col in df.columns:
    if col in _skip:
        continue
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].astype(np.float32)

print(f"✔  Rows: {len(df):,}  |  PIDs: {df['PID'].nunique():,}")
print(f"✔  Range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"✔  Dtypes after sanitisation:\n{df.dtypes.value_counts().to_string()}")


# ─────────────────────────────────────────────────────────────
# 2. BLACKMAN FIR SMOOTHING   (Dey et al. 2024)
# ─────────────────────────────────────────────────────────────
def blackman_fir_smooth(series: np.ndarray,
                         cutoff: float = 0.1,
                         order:  int   = 31) -> np.ndarray:
    """
    Zero-phase Blackman FIR low-pass filter for 1-D signals.
    Surpasses Savitzky-Golay and rolling-mean alternatives
    per Dey et al. (2024).
    """
    h       = signal.firwin(order, cutoff, window="blackman")
    padded  = np.pad(series, (order // 2, order // 2), mode="edge")
    smoothed = np.convolve(padded, h, mode="valid")
    return smoothed[:len(series)]


print("\n🔧  Applying Blackman FIR smoothing to NDVI & NDMI …")
for col in ["NDVI", "NDMI"]:
    if col in df.columns:
        df[f"{col}_smooth"] = df.groupby("PID")[col].transform(
            lambda x: blackman_fir_smooth(x.values)
        ).astype(np.float32)
        print(f"   ✔  {col}_smooth created")


# ─────────────────────────────────────────────────────────────
# 3. FEATURE COLUMN SELECTION
# ─────────────────────────────────────────────────────────────
SMOOTH_FEATURES = [c for c in ["NDVI_smooth", "NDMI_smooth"] if c in df.columns]
FEATURE_COLS    = ALL_FEATURE_COLS + SMOOTH_FEATURES
FEATURE_COLS    = [
    c for c in FEATURE_COLS
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
]
print(f"\n✔  Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
# Using 5 PIDs for a representative subset; adjust as needed
sample_pids = df["PID"].unique()[:5]
df_small    = df[df["PID"].isin(sample_pids)].copy()
train_df    = df_small[df_small["Year"] == 2022].copy()
test_df     = df_small[df_small["Year"] == 2023].copy()

print(f"\n✔  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")


# ─────────────────────────────────────────────────────────────
# 5. SCALERS
# ─────────────────────────────────────────────────────────────
def coerce_numeric(frame: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Force all listed columns to float32; non-numerics → NaN."""
    frame = frame.copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame

train_df = coerce_numeric(train_df, FEATURE_COLS)
test_df  = coerce_numeric(test_df,  FEATURE_COLS)

train_arr = train_df[FEATURE_COLS].dropna().values.astype(np.float32)

scaler_NDVI = MinMaxScaler((0, 1))
scaler_NDMI = MinMaxScaler((0, 1))
scaler_NDVI.fit(train_arr)
scaler_NDMI.fit(train_arr)

sc_NDVI_path = os.path.join(MODEL_DIR, "scaler_NDVI.pkl")
sc_NDMI_path = os.path.join(MODEL_DIR, "scaler_NDMI.pkl")
joblib.dump(scaler_NDVI, sc_NDVI_path)
joblib.dump(scaler_NDMI, sc_NDMI_path)
print(f"\n✔  Scalers saved → {MODEL_DIR}")


# ─────────────────────────────────────────────────────────────
# 6. SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────
def build_sequences(arr: np.ndarray, target_idx: int,
                    timesteps: int = TIMESTEPS,
                    horizon:   int = HORIZON):
    X, y = [], []
    for i in range(timesteps, len(arr) - horizon + 1):
        X.append(arr[i - timesteps : i, :])
        y.append(arr[i + horizon - 1, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_global_sequences(src_df, scaler, target, feat_cols):
    all_X, all_y = [], []
    for pid, pid_df in src_df.groupby("PID"):
        pid_df = pid_df.sort_values("Date").copy()
        for col in feat_cols:
            pid_df[col] = pd.to_numeric(pid_df[col], errors="coerce")
        pid_df = pid_df.dropna(subset=feat_cols)
        if len(pid_df) < TIMESTEPS + 1:
            continue
        arr  = scaler.transform(pid_df[feat_cols].values.astype(np.float32))
        tidx = feat_cols.index(target)
        X, y = build_sequences(arr, tidx)
        all_X.append(X)
        all_y.append(y)
    if not all_X:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(all_X), np.concatenate(all_y)


print("\n🔧  Building sequences (timesteps={}) …".format(TIMESTEPS))
X_tr_ndvi, y_tr_ndvi = build_global_sequences(train_df, scaler_NDVI, TARGET_NDVI, FEATURE_COLS)
X_te_ndvi, y_te_ndvi = build_global_sequences(test_df,  scaler_NDVI, TARGET_NDVI, FEATURE_COLS)
X_tr_ndmi, y_tr_ndmi = build_global_sequences(train_df, scaler_NDMI, TARGET_NDMI, FEATURE_COLS)
X_te_ndmi, y_te_ndmi = build_global_sequences(test_df,  scaler_NDMI, TARGET_NDMI, FEATURE_COLS)

print(f"✔  NDVI — train {X_tr_ndvi.shape}  test {X_te_ndvi.shape}")
print(f"✔  NDMI — train {X_tr_ndmi.shape}  test {X_te_ndmi.shape}")


# ─────────────────────────────────────────────────────────────
# 7. PERSIST ARRAYS
# ─────────────────────────────────────────────────────────────
np.save(os.path.join(ARR_DIR, "X_tr_ndvi.npy"), X_tr_ndvi)
np.save(os.path.join(ARR_DIR, "y_tr_ndvi.npy"), y_tr_ndvi)
np.save(os.path.join(ARR_DIR, "X_te_ndvi.npy"), X_te_ndvi)
np.save(os.path.join(ARR_DIR, "y_te_ndvi.npy"), y_te_ndvi)
np.save(os.path.join(ARR_DIR, "X_tr_ndmi.npy"), X_tr_ndmi)
np.save(os.path.join(ARR_DIR, "y_tr_ndmi.npy"), y_tr_ndmi)
np.save(os.path.join(ARR_DIR, "X_te_ndmi.npy"), X_te_ndmi)
np.save(os.path.join(ARR_DIR, "y_te_ndmi.npy"), y_te_ndmi)

# Save feature column list for downstream steps
feat_cols_path = os.path.join(ARR_DIR, "feature_cols.txt")
with open(feat_cols_path, "w") as fh:
    fh.write("\n".join(FEATURE_COLS))

# Save processed test_df (needed by steps 3 & 5)
test_df.to_csv(os.path.join(OUT_DIR, "test_df.csv"), index=False)
train_df.to_csv(os.path.join(OUT_DIR, "train_df.csv"), index=False)

print(f"\n✔  Arrays saved → {ARR_DIR}")
print(f"✔  Feature list saved → {feat_cols_path}")
print("\n" + "="*60)
print("  STEP 1 COMPLETE — run step2_train.py next")
print("="*60)
