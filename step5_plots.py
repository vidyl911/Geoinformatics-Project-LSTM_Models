#!/usr/bin/env python3
"""
step5_plots.py
==============
NDVI / NDMI Gap-Fill Pipeline — Step 5
  Generates all 11 publication-quality figures (300 DPI).

  Plot inventory:
    01  Training & validation loss curves
    02  Metrics comparison bar chart
    03  Density scatter (Predicted vs Observed)
    04  Advanced residual analysis (one per model)
    05  Seasonal RMSE comparison
    06  Ablation study — timestep importance
    07  Day-of-Year error profile
    08  Wilcoxon p-value heatmap
    09  Violin plots — error distributions
    10  Summary metrics table
    11  Time-series gap-fill showcase (first PID)

Run:
    python step5_plots.py

Requires outputs from Steps 1–4.
Outputs: PLOT_DIR/*.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from config import (
    OUT_DIR, MODEL_DIR, PLOT_DIR, SEED,
    PALETTE, SEASON_COLOURS, MODEL_COLOURS, MODEL_MARKERS, MODEL_LS,
    SEASON_MAP, TARGET_NDVI, TARGET_NDMI, TIMESTEPS,
)

np.random.seed(SEED)
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 0. SHARED UTILITIES
# ─────────────────────────────────────────────────────────────
def add_panel_label(ax, label, x=0.02, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=PALETTE["grid"], alpha=0.85))


def tight_save(fig, fname):
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Saved → {fname}")


def get_season(doy):
    for rng, s in SEASON_MAP:
        if doy in rng:
            return s
    return "Winter"


# ─────────────────────────────────────────────────────────────
# 1. LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────
print("\n📂  Loading artefacts …")

# Feature columns
with open(os.path.join(OUT_DIR, "arrays", "feature_cols.txt")) as fh:
    FEATURE_COLS = [l.strip() for l in fh if l.strip()]

# Training histories (one CSV per model)
HISTORY_NAMES = {
    "LSTM — NDVI":            "history_LSTM_NDVI.csv",
    "BiLSTM — NDVI":          "history_BiLSTM_NDVI.csv",
    "Smoothed-BiLSTM — NDVI": "history_SBiLSTM_NDVI.csv",
    "Attn-BiLSTM — NDVI":     "history_ABiLSTM_NDVI.csv",
    "LSTM — NDMI":            "history_LSTM_NDMI.csv",
    "BiLSTM — NDMI":          "history_BiLSTM_NDMI.csv",
    "Smoothed-BiLSTM — NDMI": "history_SBiLSTM_NDMI.csv",
    "Attn-BiLSTM — NDMI":     "history_ABiLSTM_NDMI.csv",
}
histories = {}
for name, fname in HISTORY_NAMES.items():
    p = os.path.join(MODEL_DIR, fname)
    if os.path.exists(p):
        histories[name] = pd.read_csv(p)
    else:
        print(f"  ⚠  Missing history: {fname}")

# Prediction DataFrames
MODEL_LABELS = ["LSTM", "BiLSTM", "Smoothed-BiLSTM", "Attention-BiLSTM"]
preds_ndvi, preds_ndmi = {}, {}
for lbl in MODEL_LABELS:
    fn_ndvi = os.path.join(OUT_DIR, f"pred_NDVI_{lbl.replace(' ','_')}.csv")
    fn_ndmi = os.path.join(OUT_DIR, f"pred_NDMI_{lbl.replace(' ','_')}.csv")
    if os.path.exists(fn_ndvi):
        preds_ndvi[lbl] = pd.read_csv(fn_ndvi, parse_dates=["Date"])
    if os.path.exists(fn_ndmi):
        preds_ndmi[lbl] = pd.read_csv(fn_ndmi, parse_dates=["Date"])

# Metrics
metrics_df = pd.read_csv(os.path.join(OUT_DIR, "evaluation_metrics_advanced.csv"))

# Seasonal & DoY
seasonal_dfs = {}
seasonal_csv = os.path.join(OUT_DIR, "seasonal_metrics_advanced.csv")
if os.path.exists(seasonal_csv):
    sea_all = pd.read_csv(seasonal_csv)
    for lbl in MODEL_LABELS:
        seasonal_dfs[lbl] = sea_all[sea_all["Model"] == lbl]

doy_dfs = {}
doy_csv = os.path.join(OUT_DIR, "doy_profile_advanced.csv")
if os.path.exists(doy_csv):
    doy_all = pd.read_csv(doy_csv)
    for lbl in MODEL_LABELS:
        doy_dfs[lbl] = doy_all[doy_all["Model"] == lbl]

# Wilcoxon matrices
sig_ndvi = pd.read_csv(os.path.join(OUT_DIR, "wilcoxon_pvalues_NDVI.csv"),
                        index_col=0)
sig_ndmi = pd.read_csv(os.path.join(OUT_DIR, "wilcoxon_pvalues_NDMI.csv"),
                        index_col=0)

# Ablation
ablation_df = pd.read_csv(os.path.join(OUT_DIR, "ablation_timesteps.csv"))

# test_df (for time-series showcase)
test_df = pd.read_csv(os.path.join(OUT_DIR, "test_df.csv"), parse_dates=["Date"])

print("✔  All artefacts loaded")
print("\n📊  Generating plots …")


# ─────────────────────────────────────────────────────────────
# PLOT 1 — Training loss curves
# ─────────────────────────────────────────────────────────────
if histories:
    palette_order = ([PALETTE["lstm"], PALETTE["bilstm"],
                      PALETTE["smoothed"], PALETTE["attn"]] * 2)
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), facecolor=PALETTE["bg"])
    fig.suptitle("Training & Validation Loss Convergence — All Models",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, (name, hist_df), col in zip(
        axes.flatten(), histories.items(), palette_order
    ):
        ep = range(1, len(hist_df) + 1)
        ax.plot(ep, hist_df["loss"],     color=col, lw=2.2, label="Train", alpha=0.9)
        ax.plot(ep, hist_df["val_loss"], color=col, lw=2.2, label="Val",
                linestyle="--", alpha=0.7)
        best = hist_df["val_loss"].idxmin()
        ax.axvline(best + 1, color="grey", ls=":", lw=1.2, alpha=0.7)
        ax.fill_between(ep, hist_df["loss"], hist_df["val_loss"],
                        alpha=0.08, color=col)
        ax.set_title(name.replace(" — ", "\n"), fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8, framealpha=0.8)
        min_vl = hist_df["val_loss"].min()
        ax.annotate(f"Best\n{min_vl:.5f}",
                    xy=(best + 1, min_vl),
                    xytext=(best + 1 + 0.5, min_vl * 1.15),
                    fontsize=7.5, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1))
        add_panel_label(ax, chr(65 + list(histories.keys()).index(name)))

    plt.tight_layout()
    tight_save(fig, "01_training_loss_curves.png")


# ─────────────────────────────────────────────────────────────
# PLOT 2 — Metrics comparison bars
# ─────────────────────────────────────────────────────────────
met_all = (
    metrics_df[(metrics_df["Target"] == "NDVI") &
               (metrics_df["Subset"] == "All")]
    .copy().set_index("Model")
)

fig = plt.figure(figsize=(22, 8), facecolor=PALETTE["bg"])
gs0 = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
fig.suptitle("Multi-Metric Model Comparison — NDVI (Test Set)",
             fontsize=15, fontweight="bold")

metric_meta = [
    ("RMSE", "lower",  "RMSE ↓"),
    ("MAE",  "lower",  "MAE ↓"),
    ("R²",   "higher", "R² ↑"),
    ("KGE",  "higher", "KGE ↑"),
]
for idx, (metric, direction, label) in enumerate(metric_meta):
    ax     = fig.add_subplot(gs0[idx])
    models_list = list(met_all.index)
    vals   = met_all[metric].values
    cols   = [MODEL_COLOURS.get(m, "#999") for m in models_list]
    order  = np.argsort(vals) if direction == "lower" else np.argsort(vals)[::-1]
    m_ord  = [models_list[i] for i in order]
    v_ord  = [vals[i]        for i in order]
    c_ord  = [cols[i]        for i in order]

    bars = ax.barh(range(len(m_ord)), v_ord, color=c_ord,
                   edgecolor="white", linewidth=1.2, height=0.6, alpha=0.88)
    for bar, val in zip(bars, v_ord):
        ax.text(bar.get_width() + max(v_ord) * 0.01,
                bar.get_y() + 0.3,
                f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(len(m_ord)))
    ax.set_yticklabels(m_ord, fontsize=9)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel(metric)
    ax.invert_yaxis()
    add_panel_label(ax, chr(65 + idx))

tight_save(fig, "02_metrics_comparison_bars.png")


# ─────────────────────────────────────────────────────────────
# PLOT 3 — Density scatter
# ─────────────────────────────────────────────────────────────
if preds_ndvi and preds_ndmi:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), facecolor=PALETTE["bg"])
    fig.suptitle("Predicted vs Observed — NDVI (top) & NDMI (bottom)",
                 fontsize=15, fontweight="bold")

    for row, (preds, tgt) in enumerate([(preds_ndvi, "NDVI"),
                                         (preds_ndmi, "NDMI")]):
        for col, (lbl, pdf) in enumerate(preds.items()):
            ax  = axes[row][col]
            dat = pdf.dropna(subset=["y_true", "y_pred"])
            yt, yp = dat["y_true"].values, dat["y_pred"].values
            hb = ax.hexbin(yt, yp, gridsize=35, cmap="YlOrRd",
                            mincnt=1, linewidths=0.1)
            plt.colorbar(hb, ax=ax, shrink=0.7, label="Count")
            lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            ax.plot(lims, lims, "k--", lw=1.5, label="1:1 line")
            slope, intercept, r, p, _ = stats.linregress(yt, yp)
            xfit = np.linspace(lims[0], lims[1], 100)
            ax.plot(xfit, slope * xfit + intercept,
                    color=MODEL_COLOURS.get(lbl, "#555"),
                    lw=2, label=f"y={slope:.2f}x+{intercept:.3f}")
            r2   = r2_score(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            ax.text(0.05, 0.92,
                    f"R²={r2:.3f}\nRMSE={rmse:.4f}\nn={len(yt):,}",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round", fc="white",
                               ec=PALETTE["grid"], alpha=0.85))
            ax.set_title(f"{lbl}\n({tgt})", fontsize=10, fontweight="bold")
            ax.set_xlabel(f"Observed {tgt}")
            ax.set_ylabel(f"Predicted {tgt}")
            ax.legend(fontsize=7.5)
            add_panel_label(ax, chr(65 + row * 4 + col))

    plt.tight_layout()
    tight_save(fig, "03_density_scatter_pred_obs.png")


# ─────────────────────────────────────────────────────────────
# PLOT 4 — Advanced residual analysis (per model)
# ─────────────────────────────────────────────────────────────
for lbl, pdf in preds_ndvi.items():
    dat   = pdf.dropna(subset=["y_true", "y_pred"])
    res   = dat["residual"].values
    yt    = dat["y_true"].values
    dates = pd.to_datetime(dat["Date"])
    col   = MODEL_COLOURS.get(lbl, PALETTE["lstm"])

    fig = plt.figure(figsize=(20, 13), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle(f"Advanced Residual Analysis — {lbl} (NDVI)",
                 fontsize=15, fontweight="bold")

    # A — Residuals over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(dates, res, s=4, alpha=0.25, color=col)
    roll = pd.Series(res).rolling(15, center=True).mean()
    ax1.plot(dates.values, roll.values, color="black", lw=1.8,
             label="Rolling mean (15)")
    ax1.axhline(0, color="red", lw=1.4, ls="--")
    ax1.fill_between(dates, 0, res, where=res > 0,
                     alpha=0.12, color=PALETTE["gap"], label="Over-pred")
    ax1.fill_between(dates, 0, res, where=res < 0,
                     alpha=0.12, color=PALETTE["lstm"], label="Under-pred")
    ax1.set_title("Residuals Over Time"); ax1.set_xlabel("Date")
    ax1.set_ylabel("Residual (pred − obs)")
    ax1.legend(fontsize=8); add_panel_label(ax1, "A")

    # B — Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(res, bins=55, density=True,
             color=col, edgecolor="white", alpha=0.7, label="Histogram")
    kde_x = np.linspace(res.min(), res.max(), 200)
    ax2.plot(kde_x, stats.gaussian_kde(res)(kde_x), color="black", lw=2, label="KDE")
    mu, sigma = res.mean(), res.std()
    ax2.plot(kde_x, stats.norm.pdf(kde_x, mu, sigma),
             color="red", lw=1.8, ls="--", label="Normal fit")
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=8); add_panel_label(ax2, "B")

    # C — QQ plot
    ax3 = fig.add_subplot(gs[0, 2])
    (osm, osr), (slope_qq, intercept_qq, _) = stats.probplot(res, dist="norm")
    ax3.scatter(osm, osr, s=6, alpha=0.4, color=col)
    ax3.plot(osm, slope_qq * np.array(osm) + intercept_qq,
             color="red", lw=1.8, ls="--", label="Theoretical")
    ax3.set_title("Q-Q Plot"); ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.legend(fontsize=8); add_panel_label(ax3, "C")

    # D — Residual vs Predicted
    ax4 = fig.add_subplot(gs[1, 0])
    yp_vals = dat["y_pred"].values
    ax4.scatter(yp_vals, res, s=4, alpha=0.25, color=col)
    ax4.axhline(0, color="red", lw=1.4, ls="--")
    z = np.polyfit(yp_vals, res, 1)
    xfit = np.linspace(yp_vals.min(), yp_vals.max(), 100)
    ax4.plot(xfit, np.polyval(z, xfit), color="black", lw=1.5)
    ax4.set_title("Residual vs Predicted")
    ax4.set_xlabel("Predicted NDVI"); ax4.set_ylabel("Residual")
    add_panel_label(ax4, "D")

    # E — DoY RMSE profile
    ax5 = fig.add_subplot(gs[1, 1])
    doy_df = doy_dfs.get(lbl, pd.DataFrame())
    if not doy_df.empty:
        ax5.plot(doy_df["DoY"], doy_df["RMSE"],
                 color=col, lw=2, marker="o", ms=4)
        ax5.fill_between(doy_df["DoY"], 0, doy_df["RMSE"],
                         alpha=0.15, color=col)
        for season, (start, end) in zip(
            ["Winter", "Spring", "Summer", "Autumn", "Winter"],
            [(1,59),(60,151),(152,243),(244,334),(335,365)]
        ):
            ax5.axvspan(start, end, alpha=0.06,
                        color=SEASON_COLOURS.get(season, "grey"))
    ax5.set_title("RMSE by Day-of-Year")
    ax5.set_xlabel("DoY"); ax5.set_ylabel("RMSE")
    add_panel_label(ax5, "E")

    # F — Seasonal box plots
    ax6 = fig.add_subplot(gs[1, 2])
    dat2 = dat.copy()
    dat2["season"] = dat2["DoY"].apply(get_season)
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_data  = [dat2[dat2["season"] == s]["residual"].dropna().values
                    for s in season_order]
    bp = ax6.boxplot(season_data, labels=season_order,
                     patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for patch, s in zip(bp["boxes"], season_order):
        patch.set_facecolor(SEASON_COLOURS[s])
        patch.set_alpha(0.75)
    ax6.axhline(0, color="red", lw=1.2, ls="--")
    ax6.set_title("Seasonal Residual Distributions")
    ax6.set_xlabel("Season"); ax6.set_ylabel("Residual")
    add_panel_label(ax6, "F")

    tight_save(fig, f"04_residual_analysis_{lbl.replace(' ','_').replace('-','_')}.png")


# ─────────────────────────────────────────────────────────────
# PLOT 5 — Seasonal RMSE comparison
# ─────────────────────────────────────────────────────────────
if seasonal_dfs:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor=PALETTE["bg"])
    fig.suptitle("Seasonal RMSE Comparison — NDVI (All Models)",
                 fontsize=15, fontweight="bold")

    for ax, season in zip(axes, seasons):
        model_rmse = []
        for lbl, sdf in seasonal_dfs.items():
            row  = sdf[sdf["Subset"] == season]
            rmse = row["RMSE"].values[0] if len(row) > 0 else np.nan
            model_rmse.append((lbl, rmse))
        labels_s  = [x[0] for x in model_rmse]
        values_s  = [x[1] for x in model_rmse]
        bar_cols  = [MODEL_COLOURS.get(l, "#999") for l in labels_s]
        valid_v   = [v for v in values_s if not np.isnan(v)]
        bars = ax.bar(labels_s, values_s, color=bar_cols,
                      edgecolor="white", linewidth=1.2, alpha=0.85)
        for bar, val in zip(bars, values_s):
            if not np.isnan(val) and valid_v:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + max(valid_v) * 0.01,
                        f"{val:.4f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(season, fontsize=12, fontweight="bold",
                     color=SEASON_COLOURS[season])
        ax.set_ylabel("RMSE"); ax.tick_params(axis="x", rotation=20)
        add_panel_label(ax, chr(65 + seasons.index(season)))

    plt.tight_layout()
    tight_save(fig, "05_seasonal_rmse_comparison.png")


# ─────────────────────────────────────────────────────────────
# PLOT 6 — Ablation study
# ─────────────────────────────────────────────────────────────
if not ablation_df.empty:
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["bg"])
    fig.suptitle("Timestep Ablation Study — Impact on RMSE (NDVI)",
                 fontsize=15, fontweight="bold")
    for lbl, grp in ablation_df.groupby("Model"):
        col = MODEL_COLOURS.get(lbl, "#999")
        ax.plot(grp["n_ablated"], grp["RMSE"],
                color=col, lw=2.2,
                marker=MODEL_MARKERS.get(lbl, "o"), ms=6,
                ls=MODEL_LS.get(lbl, "-"), label=lbl)
        ax.fill_between(grp["n_ablated"],
                        grp["RMSE"] * 0.98, grp["RMSE"] * 1.02,
                        alpha=0.08, color=col)
    ax.set_xlabel("Number of Ablated (Zeroed) Leading Timesteps", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks(range(0, TIMESTEPS + 1))
    ax.set_xticklabels(
        [f"t-{TIMESTEPS - i}" if i < TIMESTEPS else "None"
         for i in range(TIMESTEPS + 1)], rotation=30
    )
    add_panel_label(ax, "A")
    tight_save(fig, "06_ablation_timesteps.png")


# ─────────────────────────────────────────────────────────────
# PLOT 7 — DoY error profile
# ─────────────────────────────────────────────────────────────
if doy_dfs:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=PALETTE["bg"])
    fig.suptitle("Day-of-Year Error Profile — NDVI",
                 fontsize=15, fontweight="bold")
    for ax, (metric_col, metric_label) in zip(
        axes, [("RMSE", "RMSE"), ("ME", "Mean Error (Bias)")]
    ):
        for lbl, ddf in doy_dfs.items():
            if ddf.empty:
                continue
            col = MODEL_COLOURS.get(lbl, "#999")
            ax.plot(ddf["DoY"], ddf[metric_col],
                    color=col, lw=1.8, ls=MODEL_LS.get(lbl, "-"),
                    label=lbl, alpha=0.9)
        for season, (start, end) in zip(
            ["Winter", "Spring", "Summer", "Autumn", "Winter"],
            [(1,59),(60,151),(152,243),(244,334),(335,365)]
        ):
            ax.axvspan(start, end, alpha=0.07,
                       color=SEASON_COLOURS.get(season, "grey"))
        if metric_col == "ME":
            ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
        ax.set_xlabel("Day of Year"); ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by DoY")
        ax.legend(fontsize=9, framealpha=0.9)
        add_panel_label(ax, "A" if metric_col == "RMSE" else "B")
    plt.tight_layout()
    tight_save(fig, "07_doy_error_profile.png")


# ─────────────────────────────────────────────────────────────
# PLOT 8 — Wilcoxon heatmap
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
fig.suptitle("Wilcoxon Significance Test — p-value Matrix",
             fontsize=15, fontweight="bold")
for ax, (sig_df, tgt_lbl) in zip(axes, [(sig_ndvi, "NDVI"),
                                          (sig_ndmi, "NDMI")]):
    mask = np.eye(len(sig_df), dtype=bool)
    sns.heatmap(sig_df, annot=True, fmt=".3f", ax=ax,
                cmap="RdYlGn_r", vmin=0, vmax=0.1,
                linewidths=0.5, linecolor=PALETTE["grid"],
                mask=mask, cbar_kws={"label": "p-value"})
    ax.set_title(f"{tgt_lbl} — Pairwise Wilcoxon p-values",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
tight_save(fig, "08_wilcoxon_heatmap.png")


# ─────────────────────────────────────────────────────────────
# PLOT 9 — Violin plots
# ─────────────────────────────────────────────────────────────
if preds_ndvi and preds_ndmi:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=PALETTE["bg"])
    fig.suptitle("Error Distribution — Violin Plots (NDVI & NDMI)",
                 fontsize=15, fontweight="bold")
    for ax, (preds, tgt) in zip(axes, [(preds_ndvi, "NDVI"),
                                        (preds_ndmi, "NDMI")]):
        data_vio  = [p.dropna()["residual"].values for p in preds.values()]
        positions = list(range(len(data_vio)))
        parts = ax.violinplot(data_vio, positions=positions,
                               showmedians=True, showextrema=True)
        for pc, lbl in zip(parts["bodies"], preds.keys()):
            pc.set_facecolor(MODEL_COLOURS.get(lbl, "#999"))
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)
        ax.set_xticks(positions)
        ax.set_xticklabels(list(preds.keys()), rotation=20, fontsize=9)
        ax.axhline(0, color="red", lw=1.3, ls="--", alpha=0.7)
        ax.set_title(f"{tgt} Residuals by Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual (pred − obs)")
        add_panel_label(ax, "A" if tgt == "NDVI" else "B")
    plt.tight_layout()
    tight_save(fig, "09_violin_residuals.png")


# ─────────────────────────────────────────────────────────────
# PLOT 10 — Metrics summary table
# ─────────────────────────────────────────────────────────────
summary     = metrics_df[metrics_df["Subset"] == "All"].copy()
display_cols = ["Model", "Target", "N", "RMSE", "MAE", "R²",
                "KGE", "NSE", "PCC", "Bias"]
summary      = summary[display_cols].reset_index(drop=True)

fig, ax = plt.subplots(
    figsize=(18, len(summary) * 0.55 + 1.5), facecolor=PALETTE["bg"]
)
ax.axis("off")
fig.suptitle("Comprehensive Evaluation Metrics Summary",
             fontsize=15, fontweight="bold", y=0.98)

tbl = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(PALETTE["lstm"])
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#EEF2F7")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor(PALETTE["grid"])

tight_save(fig, "10_metrics_summary_table.png")


# ─────────────────────────────────────────────────────────────
# PLOT 11 — Time-series gap-fill showcase  (reference-style)
#
# Layout mirrors the reference figure:
#   • Grey scatter dots  = raw observed values (y_true)
#   • Thin grey line     = raw 4-week rolling mean of observed
#   • Grey shading       = raw ±1.96σ band (4-week rolling)
#   • Bold BLUE line     = Train-period 4-week rolling mean of y_pred
#     (best model, Attention-BiLSTM preferred when available)
#   • Light BLUE shading = Train ±1.96σ rolling band
#   • Vertical RED bars  = gap / disturbance events (is_gap == True)
#   • Dashed RED line    = Test-period prediction — primary model
#   • Solid GREEN line   = Test-period prediction — secondary model
#   • Shaded RED/GREEN   = ±1σ rolling CI on test predictions
#   One panel per target (NDVI top, NDMI bottom), shared x-axis.
# ─────────────────────────────────────────────────────────────
if preds_ndvi:
    # ── helper: rolling mean & 1.96σ band ───────────────────
    def _rolling_stats(series, window=28):
        s   = pd.Series(series)
        mu  = s.rolling(window, center=True, min_periods=1).mean()
        sig = s.rolling(window, center=True, min_periods=1).std().fillna(0)
        return mu.values, sig.values

    # ── choose first PID that has data in every model ────────
    candidate_pids = test_df["PID"].unique()
    first_pid = None
    for pid in candidate_pids:
        if all(not pdf[pdf["PID"] == pid].empty for pdf in preds_ndvi.values()):
            first_pid = pid
            break
    if first_pid is None:
        first_pid = candidate_pids[0]

    # ── model priority: Attention-BiLSTM > Smoothed > BiLSTM > LSTM ──
    _priority = ["Attention-BiLSTM", "Smoothed-BiLSTM", "BiLSTM", "LSTM"]
    _avail    = list(preds_ndvi.keys())
    _sorted   = sorted(_avail, key=lambda x: _priority.index(x)
                       if x in _priority else 99)
    primary_lbl   = _sorted[0]                              # bold line
    secondary_lbl = _sorted[1] if len(_sorted) > 1 else _sorted[0]

    # ── load train_df for the rolling train band ─────────────
    train_df_path = os.path.join(OUT_DIR, "train_df.csv")
    if os.path.exists(train_df_path):
        train_df_full = pd.read_csv(train_df_path, parse_dates=["Date"])
    else:
        train_df_full = pd.DataFrame()

    # ── figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1, figsize=(22, 11),
        facecolor=PALETTE["bg"],
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.suptitle(
        f"Gap-Fill Time-Series Showcase  —  PID {first_pid}\n"
        f"Primary: {primary_lbl}   |   Secondary: {secondary_lbl}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    for ax, (preds, tgt, tgt_col) in zip(
        axes,
        [
            (preds_ndvi, "NDVI", PALETTE["lstm"]),
            (preds_ndmi, "NDMI", PALETTE["bilstm"]),
        ],
    ):
        # ── pull primary PID slice ───────────────────────────
        ref_df = preds[primary_lbl]
        pid_df = ref_df[ref_df["PID"] == first_pid].copy()
        if pid_df.empty:
            ax.set_ylabel(tgt, fontsize=11)
            continue

        pid_df  = pid_df.sort_values("Date")
        dates   = pd.to_datetime(pid_df["Date"])
        y_true  = pid_df["y_true"].values

        # ── ① raw observed scatter ───────────────────────────
        ax.scatter(
            dates, y_true,
            s=18, color="#888888", alpha=0.55, zorder=4,
            marker="*", linewidths=0,
            label="raw (stars)",
        )

        # ── ② raw 4-week rolling mean & ±1.96σ band ─────────
        raw_mu, raw_sig = _rolling_stats(y_true, window=28)
        ax.plot(
            dates, raw_mu,
            color="#555555", lw=1.2, alpha=0.75, zorder=5,
            label="raw 4-w MA",
        )
        ax.fill_between(
            dates,
            raw_mu - 1.96 * raw_sig,
            raw_mu + 1.96 * raw_sig,
            alpha=0.18, color="#AAAAAA", zorder=2,
            label="raw ±1.96σ",
        )

        # ── ③ train-period rolling mean (y_pred) ─────────────
        # Use primary model's predictions as the "fitted" train curve
        tr_mu, tr_sig = _rolling_stats(pid_df["y_pred"].values, window=28)
        ax.plot(
            dates, tr_mu,
            color=tgt_col, lw=3.0, alpha=0.92, zorder=7,
            label="Train 4-w MA (y_pred)",
        )
        ax.fill_between(
            dates,
            tr_mu - 1.96 * tr_sig,
            tr_mu + 1.96 * tr_sig,
            alpha=0.12, color=tgt_col, zorder=3,
            label="Train ±1.96σ",
        )

        # ── ④ test-period predictions: primary & secondary ───
        # Split at the chronological midpoint as a proxy for
        # train/test boundary visible in the test window.
        mid_date = dates.iloc[len(dates) // 2]

        for pred_lbl, line_col, line_ls, line_label in [
            (primary_lbl,   "#D62728", "--", f"Pred (Test, {primary_lbl})"),
            (secondary_lbl, "#2CA02C", "-",  f"Pred (Test, {secondary_lbl})"),
        ]:
            sub = preds[pred_lbl]
            sub = sub[sub["PID"] == first_pid].sort_values("Date")
            if sub.empty:
                continue
            sub_dates = pd.to_datetime(sub["Date"])
            test_mask = sub_dates >= mid_date

            t_dates = sub_dates[test_mask]
            t_pred  = sub.loc[test_mask, "y_pred"].values

            if len(t_pred) < 2:
                continue

            # rolling CI for test band
            t_mu, t_sig = _rolling_stats(t_pred, window=7)

            ax.plot(
                t_dates, t_pred,
                color=line_col, lw=2.0, ls=line_ls,
                alpha=0.92, zorder=8, label=line_label,
            )
            ax.fill_between(
                t_dates,
                t_mu - t_sig,
                t_mu + t_sig,
                alpha=0.14, color=line_col, zorder=6,
            )

        # ── ⑤ gap / disturbance vertical bars ────────────────
        if "is_gap" in pid_df.columns:
            gap_dates = dates[pid_df["is_gap"].astype(bool).values]
            # width ∝ gap density (thicker when burned area proxy is large)
            for gd in gap_dates:
                ax.axvspan(
                    gd - pd.Timedelta("0.4D"),
                    gd + pd.Timedelta("0.6D"),
                    alpha=0.30, color=PALETTE["gap"], zorder=1,
                    linewidth=0,
                )
            # colour-coded bands: periods where primary > secondary
            pri_sub = preds[primary_lbl]
            pri_sub = pri_sub[pri_sub["PID"] == first_pid].sort_values("Date")
            sec_sub = preds[secondary_lbl]
            sec_sub = sec_sub[sec_sub["PID"] == first_pid].sort_values("Date")
            if not pri_sub.empty and not sec_sub.empty:
                merged = pri_sub[["Date", "y_pred"]].rename(
                    columns={"y_pred": "pri"}
                ).merge(
                    sec_sub[["Date", "y_pred"]].rename(columns={"y_pred": "sec"}),
                    on="Date", how="inner",
                )
                merged["Date"] = pd.to_datetime(merged["Date"])
                pri_gt = merged["pri"] > merged["sec"]
                sec_gt = ~pri_gt
                ax.fill_between(
                    merged["Date"], 0, 1,
                    where=pri_gt.values,
                    transform=ax.get_xaxis_transform(),
                    alpha=0.06, color="#D62728", zorder=0,
                    label=f"{primary_lbl} > {secondary_lbl}",
                )
                ax.fill_between(
                    merged["Date"], 0, 1,
                    where=sec_gt.values,
                    transform=ax.get_xaxis_transform(),
                    alpha=0.06, color="#2CA02C", zorder=0,
                    label=f"{secondary_lbl} > {primary_lbl}",
                )

        # ── axes formatting ───────────────────────────────────
        ax.set_ylabel(tgt, fontsize=11, fontweight="bold")
        ax.set_ylim(
            min(y_true.min() - 0.05, -0.02),
            max(y_true.max() + 0.08,  1.02),
        )
        ax.tick_params(axis="x", rotation=20)

        leg = ax.legend(
            fontsize=8, ncol=4, framealpha=0.92,
            loc="upper left",
            edgecolor=PALETTE["grid"],
        )
        add_panel_label(ax, "A" if tgt == "NDVI" else "B")

        # ── FIRE / gap label bar (width proportional to gap count) ──
        if "is_gap" in pid_df.columns and pid_df["is_gap"].any():
            # annotate the legend with gap bar description
            ax.annotate(
                "▐ FIRE (width × Burned area)",
                xy=(0.01, 0.04), xycoords="axes fraction",
                fontsize=8, color=PALETTE["gap"], fontweight="bold",
            )

    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    tight_save(fig, "11_timeseries_gapfill_showcase.png")


print("\n" + "="*70)
print(f"  ✅  ALL PLOTS COMPLETE — saved to: {PLOT_DIR}")
print("="*70)
