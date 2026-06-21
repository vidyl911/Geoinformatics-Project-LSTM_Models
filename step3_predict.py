#!/usr/bin/env python3
"""
step3_predict.py
================
NDVI / NDMI Gap-Fill Pipeline — Step 3
  • Load saved models (all 4 architectures × 2 targets)
  • Run predictions on the test set
  • Perform inverse-scaling to original value range
  • Compute 12-metric evaluation suite
  • Save per-model prediction CSVs
  • Save comprehensive metrics CSV
  • Seasonal breakdown + Wilcoxon significance matrix

Run:
    python step3_predict.py

Outputs (written to OUT_DIR):
    pred_NDVI_<model>.csv
    pred_NDMI_<model>.csv
    evaluation_metrics_advanced.csv
    seasonal_metrics_advanced.csv
    wilcoxon_pvalues_NDVI.csv
    wilcoxon_pvalues_NDMI.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr, spearmanr
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, explained_variance_score)
import joblib
import tensorflow as tf

from config import (
    OUT_DIR, MODEL_DIR, SEED,
    TARGET_NDVI, TARGET_NDMI, TIMESTEPS, HORIZON,
    SEASON_MAP,
)

tf.random.set_seed(SEED)
np.random.seed(SEED)

ARR_DIR = os.path.join(OUT_DIR, "arrays")


# ─────────────────────────────────────────────────────────────
# 1. LOAD EVERYTHING PRODUCED BY STEPS 1 & 2
# ─────────────────────────────────────────────────────────────
print("\n📂  Loading artefacts from Steps 1 & 2 …")

with open(os.path.join(ARR_DIR, "feature_cols.txt")) as fh:
    FEATURE_COLS = [l.strip() for l in fh if l.strip()]
print(f"✔  Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

scaler_NDVI = joblib.load(os.path.join(MODEL_DIR, "scaler_NDVI.pkl"))
scaler_NDMI = joblib.load(os.path.join(MODEL_DIR, "scaler_NDMI.pkl"))
print("✔  Scalers loaded")

test_df = pd.read_csv(
    os.path.join(OUT_DIR, "test_df.csv"), parse_dates=["Date"]
)
# Re-coerce — CSV round-trips may lose dtype
for col in FEATURE_COLS:
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").astype(np.float32)
if "DoY" not in test_df.columns:
    test_df["DoY"] = test_df["Date"].dt.dayofyear
print(f"✔  test_df loaded: {len(test_df):,} rows")

MODEL_FILES = {
    "LSTM":             ("LSTM_NDVI.keras",    "LSTM_NDMI.keras"),
    "BiLSTM":           ("BiLSTM_NDVI.keras",  "BiLSTM_NDMI.keras"),
    "Smoothed-BiLSTM":  ("SBiLSTM_NDVI.keras", "SBiLSTM_NDMI.keras"),
    "Attention-BiLSTM": ("ABiLSTM_NDVI.keras", "ABiLSTM_NDMI.keras"),
}

models_ndvi, models_ndmi = {}, {}
for lbl, (fn_ndvi, fn_ndmi) in MODEL_FILES.items():
    p_ndvi = os.path.join(MODEL_DIR, fn_ndvi)
    p_ndmi = os.path.join(MODEL_DIR, fn_ndmi)
    models_ndvi[lbl] = tf.keras.models.load_model(p_ndvi)
    models_ndmi[lbl] = tf.keras.models.load_model(p_ndmi)
    print(f"✔  {lbl} models loaded")


# ─────────────────────────────────────────────────────────────
# 2. SHARED HELPERS
# ─────────────────────────────────────────────────────────────
def build_sequences(arr: np.ndarray, target_idx: int):
    X, y = [], []
    for i in range(TIMESTEPS, len(arr) - HORIZON + 1):
        X.append(arr[i - TIMESTEPS : i, :])
        y.append(arr[i + HORIZON - 1, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def inv_scale(y_sc: np.ndarray, scaler, feat_cols: list, target: str):
    """Inverse-transform a scaled 1-D target vector."""
    dummy = np.zeros((len(y_sc), len(feat_cols)), dtype=np.float32)
    tidx  = feat_cols.index(target)
    dummy[:, tidx] = y_sc.ravel()
    return scaler.inverse_transform(dummy)[:, tidx]


def predict_gapfill(model, scaler, src_df, feat_cols, target, label):
    """Predict over all PIDs; returns a tidy DataFrame."""
    results = []
    for pid, pdf in src_df.groupby("PID"):
        pdf = pdf.sort_values("Date").copy()
        for col in feat_cols:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
        valid = pdf.dropna(subset=feat_cols)
        if len(valid) < TIMESTEPS + 1:
            continue
        arr       = scaler.transform(valid[feat_cols].values.astype(np.float32))
        tidx      = feat_cols.index(target)
        X, y_true = build_sequences(arr, tidx)
        y_pred_sc = model.predict(X, verbose=0).ravel()
        y_pred    = inv_scale(y_pred_sc, scaler, feat_cols, target)
        y_true_v  = inv_scale(y_true,   scaler, feat_cols, target)
        dates     = valid["Date"].values[TIMESTEPS:]
        doys      = valid["DoY"].values[TIMESTEPS:]

        flag_col  = f"{target}_flat_flag"
        if flag_col in valid.columns:
            is_gap = valid[flag_col].values[TIMESTEPS:].astype(bool)
        else:
            is_gap = np.zeros(len(dates), dtype=bool)
        results.append(pd.DataFrame({
            "PID": pid, "Date": dates, "DoY": doys,
            "y_true": y_true_v, "y_pred": y_pred,
            "is_gap": is_gap, "split": label, "target": target,
            "residual": y_pred - y_true_v,
        }))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 3. GENERATE PREDICTIONS
# ─────────────────────────────────────────────────────────────
print("\n🔮  Generating predictions …")
preds_ndvi, preds_ndmi = {}, {}
for lbl in MODEL_FILES:
    print(f"  {lbl}", end=" ", flush=True)
    preds_ndvi[lbl] = predict_gapfill(
        models_ndvi[lbl], scaler_NDVI, test_df, FEATURE_COLS, TARGET_NDVI, "test"
    )
    preds_ndmi[lbl] = predict_gapfill(
        models_ndmi[lbl], scaler_NDMI, test_df, FEATURE_COLS, TARGET_NDMI, "test"
    )
    print("✔")

# Persist per-model CSVs
for lbl, pdf in preds_ndvi.items():
    pdf.to_csv(
        os.path.join(OUT_DIR, f"pred_NDVI_{lbl.replace(' ','_')}.csv"),
        index=False,
    )
for lbl, pdf in preds_ndmi.items():
    pdf.to_csv(
        os.path.join(OUT_DIR, f"pred_NDMI_{lbl.replace(' ','_')}.csv"),
        index=False,
    )
print("✔  Prediction CSVs saved")


# ─────────────────────────────────────────────────────────────
# 4. 12-METRIC EVALUATION
# ─────────────────────────────────────────────────────────────
def compute_all_metrics(pred_df, model_name, target_name,
                         gap_only=False, subset_label="All"):
    if gap_only:
        mask = pred_df["is_gap"].astype(bool)
        data = pred_df[mask]
    else:
        data = pred_df
    data = data.dropna(subset=["y_true", "y_pred"])
    if len(data) < 2:
        return {}   # not enough data — skip silently
    yt, yp = data["y_true"].values, data["y_pred"].values
    res    = yp - yt

    rmse  = np.sqrt(mean_squared_error(yt, yp))
    mae   = mean_absolute_error(yt, yp)
    r2    = r2_score(yt, yp)
    bias  = res.mean()
    mape  = np.mean(np.abs(res / (np.abs(yt) + 1e-9))) * 100
    evs   = explained_variance_score(yt, yp)
    pcc,_ = pearsonr(yt, yp)
    scc,_ = spearmanr(yt, yp)
    sigma_obs  = yt.std()
    sigma_pred = yp.std()
    crmsd = np.sqrt(np.mean((yp - yp.mean() - (yt - yt.mean()))**2))
    nse   = 1 - np.sum(res**2) / np.sum((yt - yt.mean())**2)
    kge   = 1 - np.sqrt(
        (pcc - 1)**2
        + (sigma_pred / sigma_obs - 1)**2
        + (yp.mean() / yt.mean() - 1)**2
    )
    return {
        "Model": model_name, "Target": target_name,
        "Subset": "Gaps" if gap_only else subset_label,
        "N": len(yt),
        "RMSE":  round(rmse, 5),  "MAE":   round(mae, 5),
        "R²":    round(r2, 4),    "Bias":  round(bias, 5),
        "MAPE%": round(mape, 3),  "EVS":   round(evs, 4),
        "PCC":   round(pcc, 4),   "SCC":   round(scc, 4),
        "CRMSD": round(crmsd, 5), "NSE":   round(nse, 4),
        "KGE":   round(kge, 4),
        "σ_obs": round(sigma_obs, 5), "σ_pred": round(sigma_pred, 5),
    }


rows = []
for lbl, pdf in preds_ndvi.items():
    rows += [
        compute_all_metrics(pdf, lbl, "NDVI", False),
        compute_all_metrics(pdf, lbl, "NDVI", True),
    ]
for lbl, pdf in preds_ndmi.items():
    rows += [
        compute_all_metrics(pdf, lbl, "NDMI", False),
        compute_all_metrics(pdf, lbl, "NDMI", True),
    ]
metrics_df = pd.DataFrame([r for r in rows if r])
metrics_df.to_csv(
    os.path.join(OUT_DIR, "evaluation_metrics_advanced.csv"), index=False
)
print("\n" + "="*80)
print("  COMPREHENSIVE EVALUATION METRICS  (All-data subset)")
print("="*80)
print(metrics_df[metrics_df["Subset"] == "All"].to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 5. SEASONAL BREAKDOWN
# ─────────────────────────────────────────────────────────────
def get_season(doy):
    for rng, s in SEASON_MAP:
        if doy in rng:
            return s
    return "Winter"


def seasonal_breakdown(pred_df, model_name):
    df_ = pred_df.dropna(subset=["y_true", "y_pred"]).copy()
    df_["season"] = df_["DoY"].apply(get_season)
    rows = []
    for s, grp in df_.groupby("season"):
        if len(grp) < 2:
            continue
        rows.append(compute_all_metrics(grp, model_name, "", False, s))
    return pd.DataFrame(rows)


def doy_profile(pred_df, model_name, window=7):
    """Rolling DoY performance profile (Farbo et al. 2024, Fig. 9 extended)."""
    df_ = pred_df.dropna(subset=["y_true", "y_pred"]).copy().sort_values("DoY")
    results = []
    for doy in range(1, 366, window):
        sub = df_[(df_["DoY"] >= doy) & (df_["DoY"] < doy + window)]
        if len(sub) < 5:
            continue
        results.append({
            "DoY":   doy + window // 2,
            "RMSE":  np.sqrt(mean_squared_error(sub["y_true"], sub["y_pred"])),
            "ME":    (sub["y_pred"] - sub["y_true"]).mean(),
            "Model": model_name,
        })
    return pd.DataFrame(results)


seasonal_dfs = {lbl: seasonal_breakdown(pdf, lbl) for lbl, pdf in preds_ndvi.items()}
doy_dfs      = {lbl: doy_profile(pdf, lbl) for lbl, pdf in preds_ndvi.items()}

pd.concat(seasonal_dfs.values()).to_csv(
    os.path.join(OUT_DIR, "seasonal_metrics_advanced.csv"), index=False
)
pd.concat(doy_dfs.values()).to_csv(
    os.path.join(OUT_DIR, "doy_profile_advanced.csv"), index=False
)
print("✔  Seasonal & DoY CSVs saved")


# ─────────────────────────────────────────────────────────────
# 6. WILCOXON SIGNIFICANCE TESTS
# ─────────────────────────────────────────────────────────────
def wilcoxon_matrix(preds_dict):
    models = list(preds_dict.keys())
    n      = len(models)
    pmat   = np.ones((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                continue
            r1 = preds_dict[m1].dropna()["residual"].abs().values
            r2 = preds_dict[m2].dropna()["residual"].abs().values
            mn = min(len(r1), len(r2))
            if mn < 10:
                continue
            try:
                _, p = wilcoxon(r1[:mn], r2[:mn])
                pmat[i, j] = p
            except Exception:
                pass
    return pd.DataFrame(pmat, index=models, columns=models)


sig_ndvi = wilcoxon_matrix(preds_ndvi)
sig_ndmi = wilcoxon_matrix(preds_ndmi)
sig_ndvi.to_csv(os.path.join(OUT_DIR, "wilcoxon_pvalues_NDVI.csv"))
sig_ndmi.to_csv(os.path.join(OUT_DIR, "wilcoxon_pvalues_NDMI.csv"))
print("✔  Wilcoxon significance matrices saved")


print("\n" + "="*60)
print("  ✅  STEP 3 COMPLETE — run step4_ablation.py next")
print("="*60)
