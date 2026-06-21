# NDVI / NDMI Gap-Fill Pipeline
### Advanced LSTM · BiLSTM · Smoothed-BiLSTM · Attention-BiLSTM

---

## File structure

```
ndvi_pipeline/
├── config.py              # All shared constants, paths, colours, rcParams
│
├── step1_data_prep.py     # Load CSV → smooth → scale → build sequences
├── step2_train.py         # Define & train all 8 models (4 arch × 2 targets)
├── step3_predict.py       # Predict, inverse-scale, compute 12 metrics
├── step4_ablation.py      # Timestep ablation study
├── step5_plots.py         # All 11 publication-quality figures
│
└── run_all.py             # Orchestrator — runs steps 1-5 in sequence
```

---

## Quick start

### Run everything
```bash
python run_all.py
```

### Resume from a specific step (e.g. after changing EPOCHS)
```bash
python run_all.py --from 2   # re-trains and regenerates plots
python run_all.py --from 3   # keeps trained models, re-runs predictions
python run_all.py --from 5   # re-generates plots only
```

### Run a single step
```bash
python run_all.py --only 5
```

### Run steps individually
```bash
python step1_data_prep.py
python step2_train.py
python step3_predict.py
python step4_ablation.py
python step5_plots.py
```

---

## Configuration (`config.py`)

| Variable        | Default                            | Notes                                     |
|-----------------|------------------------------------|-------------------------------------------|
| `DATA_PATH`     | `F:/…/cleaned_full_dataset.csv`    | Change to your actual CSV path            |
| `OUT_DIR`       | `F:/…/lstm_results_advanced`       | All outputs land here                     |
| `EPOCHS`        | `50`                               | Lower to `3` for a quick smoke-test       |
| `TIMESTEPS`     | `7`                                | Lookback window (days)                    |
| `BATCH`         | `32`                               | Mini-batch size                           |
| `VAL_SPLIT`     | `0.15`                             | Fraction of train used for validation     |

---

## Outputs

```
OUT_DIR/
├── arrays/               ← .npy sequence arrays + feature_cols.txt
├── models/               ← 8 × .keras models + history CSVs + scalers
├── plots/                ← 11 × 300 DPI figures
│
├── test_df.csv           ← pre-processed test split
├── train_df.csv          ← pre-processed train split
├── pred_NDVI_*.csv       ← per-model predictions
├── pred_NDMI_*.csv
├── evaluation_metrics_advanced.csv   ← 12-metric comparison table
├── seasonal_metrics_advanced.csv     ← per-season breakdown
├── doy_profile_advanced.csv          ← rolling DoY RMSE & ME
├── wilcoxon_pvalues_NDVI.csv         ← pairwise significance matrix
└── wilcoxon_pvalues_NDMI.csv
```

---

## Model architectures

| Label              | Params (approx) | Key novelty                                      |
|--------------------|-----------------|--------------------------------------------------|
| LSTM               | ~170 K          | Two-layer stacked LSTM baseline                  |
| BiLSTM             | ~340 K          | Two-layer bidirectional LSTM                     |
| Smoothed-BiLSTM    | ~860 K          | Three-layer BiLSTM on Blackman-smoothed features |
| Attention-BiLSTM   | ~420 K          | Multi-head self-attention + BiLSTM encoder       |

---

## Dependencies

```
tensorflow >= 2.14
scikit-learn
scipy
pandas
numpy
matplotlib
seaborn
joblib
```
