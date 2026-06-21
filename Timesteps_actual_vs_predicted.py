import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# 1. Load the Excel file
# =====================================================
df = pd.read_excel('7Timesteps_Actual_vs_Predicted.xlsx')

# =====================================================
# 2. Rename columns
# =====================================================
df.columns = [
    'Time Step', 'Date (2023)',
    'Actual NDVI', 'Pred_NDVI_LSTM', 'Pred_NDVI_BiLSTM',
    'Pred_NDVI_Smoothed-BiLSTM', 'Pred_NDVI_Attention-BiLSTM',
    'Actual NDMI', 'Pred_NDMI_LSTM', 'Pred_NDMI_BiLSTM',
    'Pred_NDMI_Smoothed-BiLSTM', 'Pred_NDMI_Attention-BiLSTM'
]

# =====================================================
# 3. Convert date column safely
# =====================================================
df['Date (2023)'] = pd.to_datetime(
    df['Date (2023)'],
    errors='coerce'
)

# Optional: Check for invalid dates
if df['Date (2023)'].isna().any():
    print("Warning: Some dates could not be parsed.")

# =====================================================
# 4. Create x-axis labels
# Example:
# t-7
# [01-03]
# =====================================================
df['X_Label'] = (
    df['Time Step'].astype(str)
    + '\n['
    + df['Date (2023)'].dt.strftime('%d-%m')
    + ']'
)

x_labels = df['X_Label'].tolist()
x_ticks = np.arange(len(df))

# =====================================================
# 5. Plot styling
# =====================================================
colors = {
    'Actual': '#2C3E50',
    'LSTM': '#E67E22',
    'BiLSTM': '#27AE60',
    'Smoothed-BiLSTM': '#8E44AD',
    'Attention-BiLSTM': '#C0392B'
}

markers = {
    'Actual': 'o',
    'LSTM': 's',
    'BiLSTM': '^',
    'Smoothed-BiLSTM': 'D',
    'Attention-BiLSTM': '*'
}

# =====================================================
# NDVI Plot
# =====================================================
fig1, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(
    x_ticks,
    df['Actual NDVI'],
    color=colors['Actual'],
    marker=markers['Actual'],
    linewidth=2.5,
    markersize=9,
    label='Actual NDVI',
    zorder=5
)

ax1.plot(
    x_ticks,
    df['Pred_NDVI_LSTM'],
    color=colors['LSTM'],
    marker=markers['LSTM'],
    linewidth=2,
    markersize=7,
    linestyle='--',
    label='LSTM'
)

ax1.plot(
    x_ticks,
    df['Pred_NDVI_BiLSTM'],
    color=colors['BiLSTM'],
    marker=markers['BiLSTM'],
    linewidth=2,
    markersize=7,
    linestyle='--',
    label='BiLSTM'
)

ax1.plot(
    x_ticks,
    df['Pred_NDVI_Smoothed-BiLSTM'],
    color=colors['Smoothed-BiLSTM'],
    marker=markers['Smoothed-BiLSTM'],
    linewidth=2,
    markersize=7,
    linestyle='--',
    label='Smoothed-BiLSTM'
)

ax1.plot(
    x_ticks,
    df['Pred_NDVI_Attention-BiLSTM'],
    color=colors['Attention-BiLSTM'],
    marker=markers['Attention-BiLSTM'],
    linewidth=2,
    markersize=8,
    linestyle='--',
    label='Attention-BiLSTM'
)

ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_labels, fontsize=10)

ax1.set_ylabel(
    'NDVI Value',
    fontsize=12,
    fontweight='bold'
)

ax1.set_title(
    'NDVI: Actual vs Model Predictions (t-7 to t)',
    fontsize=14,
    fontweight='bold'
)

ax1.legend(
    loc='best',
    frameon=True,
    framealpha=0.9
)

ax1.grid(
    True,
    linestyle=':',
    alpha=0.6
)

plt.tight_layout()

plt.savefig(
    'NDVI_Actual_vs_Predicted.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =====================================================
# NDMI Plot
# =====================================================
fig2, ax2 = plt.subplots(figsize=(12, 7))

ax2.plot(
    x_ticks,
    df['Actual NDMI'],
    color=colors['Actual'],
    marker=markers['Actual'],
    linewidth=2.5,
    markersize=9,
    label='Actual NDMI',
    zorder=5
)

ax2.plot(
    x_ticks,
    df['Pred_NDMI_LSTM'],
    color=colors['LSTM'],
    marker=markers['LSTM'],
    linewidth=2,
    markersize=7,
    linestyle='--',
    label='LSTM'
)

ax2.plot(
    x_ticks,
    df['Pred_NDMI_BiLSTM'],
    color=colors['BiLSTM'],
    marker=markers['BiLSTM'],
    linewidth=2,
    markersize=7,
    linestyle='--',
    label='BiLSTM'
)

ax2.plot(
    x_ticks,
    df['Pred_NDMI_Smoothed-BiLSTM'],
    color=colors['Smoothed-BiLSTM'],
    marker=markers['Smoothed-BiLSTM'],
    linewidth=2,
    markersize=7,
    linestyle='--',
    label='Smoothed-BiLSTM'
)

ax2.plot(
    x_ticks,
    df['Pred_NDMI_Attention-BiLSTM'],
    color=colors['Attention-BiLSTM'],
    marker=markers['Attention-BiLSTM'],
    linewidth=2,
    markersize=8,
    linestyle='--',
    label='Attention-BiLSTM'
)

ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_labels, fontsize=10)

ax2.set_ylabel(
    'NDMI Value',
    fontsize=12,
    fontweight='bold'
)

ax2.set_title(
    'NDMI: Actual vs Model Predictions (t-7 to t)',
    fontsize=14,
    fontweight='bold'
)

ax2.legend(
    loc='best',
    frameon=True,
    framealpha=0.9
)

ax2.grid(
    True,
    linestyle=':',
    alpha=0.6
)

plt.tight_layout()

plt.savefig(
    'NDMI_Actual_vs_Predicted.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Plots saved successfully:")
print(" - NDVI_Actual_vs_Predicted.png")
print(" - NDMI_Actual_vs_Predicted.png")