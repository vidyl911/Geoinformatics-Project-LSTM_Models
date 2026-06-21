#!/usr/bin/env python3
"""
step2_train.py
==============
NDVI / NDMI Gap-Fill Pipeline — Step 2
  • Load pre-built sequences from Step 1
  • Define all four model architectures
      – LSTM
      – BiLSTM
      – Smoothed-BiLSTM   (Dey et al. 2024, upgraded)
      – Attention-BiLSTM  (novel)
  • Train every model for both NDVI and NDMI targets
  • Save training history CSVs and .keras model files

Run:
    python step2_train.py

Outputs (written to MODEL_DIR):
    LSTM_NDVI.keras   BiLSTM_NDVI.keras
    SBiLSTM_NDVI.keras  ABiLSTM_NDVI.keras
    LSTM_NDMI.keras   BiLSTM_NDMI.keras
    SBiLSTM_NDMI.keras  ABiLSTM_NDMI.keras
    history_<name>.csv  (one per model)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout,
    Input, BatchNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, LayerNormalization, Add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from config import (
    OUT_DIR, MODEL_DIR, SEED,
    TIMESTEPS, BATCH, EPOCHS, VAL_SPLIT,
)

tf.random.set_seed(SEED)
np.random.seed(SEED)

ARR_DIR = os.path.join(OUT_DIR, "arrays")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"✔  TensorFlow: {tf.__version__}")
print(f"✔  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ─────────────────────────────────────────────────────────────
# 1. LOAD SEQUENCES
# ─────────────────────────────────────────────────────────────
def load_arrays():
    X_tr_ndvi = np.load(os.path.join(ARR_DIR, "X_tr_ndvi.npy"))
    y_tr_ndvi = np.load(os.path.join(ARR_DIR, "y_tr_ndvi.npy"))
    X_te_ndvi = np.load(os.path.join(ARR_DIR, "X_te_ndvi.npy"))
    y_te_ndvi = np.load(os.path.join(ARR_DIR, "y_te_ndvi.npy"))
    X_tr_ndmi = np.load(os.path.join(ARR_DIR, "X_tr_ndmi.npy"))
    y_tr_ndmi = np.load(os.path.join(ARR_DIR, "y_tr_ndmi.npy"))
    X_te_ndmi = np.load(os.path.join(ARR_DIR, "X_te_ndmi.npy"))
    y_te_ndmi = np.load(os.path.join(ARR_DIR, "y_te_ndmi.npy"))
    return (X_tr_ndvi, y_tr_ndvi, X_te_ndvi, y_te_ndvi,
            X_tr_ndmi, y_tr_ndmi, X_te_ndmi, y_te_ndmi)


print("\n📂  Loading sequences …")
(X_tr_ndvi, y_tr_ndvi, X_te_ndvi, y_te_ndvi,
 X_tr_ndmi, y_tr_ndmi, X_te_ndmi, y_te_ndmi) = load_arrays()

N_FEAT = X_tr_ndvi.shape[2]
print(f"✔  NDVI train {X_tr_ndvi.shape}  |  NDMI train {X_tr_ndmi.shape}")
print(f"✔  n_features: {N_FEAT}")


# ─────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────
def build_lstm(ts, nf, u1=128, u2=64, dr=0.2, lr=1e-3, name="LSTM"):
    """Two-layer stacked LSTM baseline."""
    inp = Input((ts, nf))
    x   = LSTM(u1, return_sequences=True)(inp)
    x   = Dropout(dr)(x)
    x   = LSTM(u2)(x)
    x   = Dropout(dr)(x)
    x   = BatchNormalization()(x)
    x   = Dense(32, "relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name=name)
    m.compile(optimizer=Adam(lr), loss=MeanSquaredError())
    return m


def build_bilstm(ts, nf, u1=128, u2=64, dr=0.2, lr=1e-3, name="BiLSTM"):
    """Two-layer Bidirectional LSTM."""
    inp = Input((ts, nf))
    x   = Bidirectional(LSTM(u1, return_sequences=True))(inp)
    x   = Dropout(dr)(x)
    x   = Bidirectional(LSTM(u2))(x)
    x   = Dropout(dr)(x)
    x   = BatchNormalization()(x)
    x   = Dense(32, "relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name=name)
    m.compile(optimizer=Adam(lr), loss=MeanSquaredError())
    return m


def build_smoothed_bilstm(ts, nf, u1=180, u2=80, dr=0.2, lr=1e-3,
                           name="Smoothed_BiLSTM"):
    """
    Smoothed-BiLSTM: three-layer BiLSTM on Blackman-smoothed features.
    Inspired by Dey et al. (2024) Smoothed-LSTM; upgraded to BiLSTM.
    """
    inp = Input((ts, nf))
    x   = Bidirectional(LSTM(u1, return_sequences=True))(inp)
    x   = Dropout(dr)(x)
    x   = Bidirectional(LSTM(u2, return_sequences=True))(x)
    x   = Dropout(dr)(x)
    x   = Bidirectional(LSTM(u2 // 2))(x)
    x   = BatchNormalization()(x)
    x   = Dense(64, "relu")(x)
    x   = Dropout(dr / 2)(x)
    x   = Dense(32, "relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name=name)
    m.compile(optimizer=Adam(lr), loss=MeanSquaredError())
    return m


def build_attention_bilstm(ts, nf, u1=128, heads=4, dr=0.2, lr=1e-3,
                            name="Attention_BiLSTM"):
    """
    Attention-BiLSTM: Novel architecture combining multi-head
    self-attention with bidirectional LSTM (original contribution).
    """
    inp  = Input((ts, nf))
    enc  = Bidirectional(LSTM(u1, return_sequences=True))(inp)
    enc  = Dropout(dr)(enc)
    attn = MultiHeadAttention(num_heads=heads,
                               key_dim=u1 // heads)(enc, enc)
    attn = Dropout(dr)(attn)
    x    = LayerNormalization()(Add()([enc, attn]))
    x    = GlobalAveragePooling1D()(x)
    x    = BatchNormalization()(x)
    x    = Dense(64, "relu")(x)
    x    = Dropout(dr / 2)(x)
    x    = Dense(32, "relu")(x)
    out  = Dense(1)(x)
    m    = Model(inp, out, name=name)
    m.compile(optimizer=Adam(lr), loss=MeanSquaredError())
    return m


# Print architecture summaries
for arch_name, builder in [
    ("LSTM",             lambda: build_lstm(TIMESTEPS, N_FEAT)),
    ("BiLSTM",           lambda: build_bilstm(TIMESTEPS, N_FEAT)),
    ("Smoothed_BiLSTM",  lambda: build_smoothed_bilstm(TIMESTEPS, N_FEAT)),
    ("Attention_BiLSTM", lambda: build_attention_bilstm(TIMESTEPS, N_FEAT)),
]:
    m = builder()
    print(f"\n{'─'*55}\n  {arch_name}: {m.count_params():,} params\n{'─'*55}")
    del m


# ─────────────────────────────────────────────────────────────
# 3. TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────
def safe_array(arr: np.ndarray) -> np.ndarray:
    """Cast to float32 and replace non-finite values with 0."""
    arr = np.array(arr, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def train_model(model, X_tr, y_tr, name: str):
    """Train a model and save the .keras file + history CSV."""
    print(f"\n  ▶  {name} …", end=" ", flush=True)
    X_tr = safe_array(X_tr)
    y_tr = safe_array(y_tr)

    history = model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_split=VAL_SPLIT,
        verbose=0,
    )
    best_val = min(history.history["val_loss"])
    print(f"best val_loss = {best_val:.6f}")

    # Save model
    save_path = os.path.join(MODEL_DIR, f"{name}.keras")
    try:
        model.save(save_path)
        print(f"       model  → {save_path}")
    except Exception as exc:
        print(f"\n       ⚠  Could not save {name}.keras: {exc}")

    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_df["epoch"] = range(1, len(hist_df) + 1)
    hist_path = os.path.join(MODEL_DIR, f"history_{name}.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"       history → {hist_path}")

    return history


# ─────────────────────────────────────────────────────────────
# 4. TRAIN ALL MODELS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  🏋️   Training NDVI models")
print("="*60)

m_lstm_ndvi  = build_lstm(TIMESTEPS, N_FEAT, name="LSTM_NDVI")
m_bil_ndvi   = build_bilstm(TIMESTEPS, N_FEAT, name="BiLSTM_NDVI")
m_smo_ndvi   = build_smoothed_bilstm(TIMESTEPS, N_FEAT, name="SBiLSTM_NDVI")
m_att_ndvi   = build_attention_bilstm(TIMESTEPS, N_FEAT, name="ABiLSTM_NDVI")

h_lstm_ndvi  = train_model(m_lstm_ndvi,  X_tr_ndvi, y_tr_ndvi, "LSTM_NDVI")
h_bil_ndvi   = train_model(m_bil_ndvi,   X_tr_ndvi, y_tr_ndvi, "BiLSTM_NDVI")
h_smo_ndvi   = train_model(m_smo_ndvi,   X_tr_ndvi, y_tr_ndvi, "SBiLSTM_NDVI")
h_att_ndvi   = train_model(m_att_ndvi,   X_tr_ndvi, y_tr_ndvi, "ABiLSTM_NDVI")

print("\n" + "="*60)
print("  🏋️   Training NDMI models")
print("="*60)

m_lstm_ndmi  = build_lstm(TIMESTEPS, N_FEAT, name="LSTM_NDMI")
m_bil_ndmi   = build_bilstm(TIMESTEPS, N_FEAT, name="BiLSTM_NDMI")
m_smo_ndmi   = build_smoothed_bilstm(TIMESTEPS, N_FEAT, name="SBiLSTM_NDMI")
m_att_ndmi   = build_attention_bilstm(TIMESTEPS, N_FEAT, name="ABiLSTM_NDMI")

h_lstm_ndmi  = train_model(m_lstm_ndmi,  X_tr_ndmi, y_tr_ndmi, "LSTM_NDMI")
h_bil_ndmi   = train_model(m_bil_ndmi,   X_tr_ndmi, y_tr_ndmi, "BiLSTM_NDMI")
h_smo_ndmi   = train_model(m_smo_ndmi,   X_tr_ndmi, y_tr_ndmi, "SBiLSTM_NDMI")
h_att_ndmi   = train_model(m_att_ndmi,   X_tr_ndmi, y_tr_ndmi, "ABiLSTM_NDMI")

print("\n" + "="*60)
print("  ✅  STEP 2 COMPLETE — run step3_predict.py next")
print("="*60)
