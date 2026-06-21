# 🌿 LSTM-Based Prediction of Vegetation Indices (NDVI & NDMI)

Welcome to the official repository for the Geoinformatics Engineering MSc project at **Politecnico di Milano**. 

This project addresses the critical challenge of data gaps in environmental sensor networks by implementing an end-to-end deep learning pipeline to predict and gap-fill vegetation indices. Using a comprehensive set of 21 meteorological, topographic, and temporal predictors, we evaluate four Long Short-Term Memory (LSTM) architectures for the simultaneous reconstruction of the Normalized Difference Vegetation Index (NDVI) and Normalized Difference Moisture Index (NDMI) across the topographically complex Lombardy region, Italy.

### 🚀 Key Highlights
* **Multi-Architecture Comparison:** Evaluates Standard LSTM, Bidirectional LSTM (BiLSTM), Blackman FIR-filtered Smoothed-BiLSTM, and a novel Attention-BiLSTM.
* **Dual-Target Prediction:** Simultaneous gap-filling for both vegetation greenness (NDVI) and canopy moisture (NDMI).
* **Comprehensive Feature Engineering:** Utilizes 21 predictors including cyclic temporal encodings, rolling statistics, and autoregressive lag features.
* **Rigorous Evaluation:** Features a 12-metric evaluation framework, seasonal/topographic stratification, timestep ablation studies, and Wilcoxon signed-rank statistical testing.
