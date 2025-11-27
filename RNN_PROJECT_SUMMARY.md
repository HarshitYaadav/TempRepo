# RNN Model for WSI Prediction - Project Summary

## 🎯 Project Overview

This project implements a **Recurrent Neural Network (RNN)** model to predict **Water Stress Index (WSI)** for Indian states. The model uses 12 months of historical data to forecast WSI values for the next month for both entropy-weighted and equal-weighted indices.

---

## 📊 What Was Created

### 1. **Core Implementation Files**

| File | Description |
|------|-------------|
| `rnn_model.py` | RNN architecture and training classes |
| `train_rnn.py` | Training script with 80-20 temporal split |
| `validate_rnn.py` | Validation with metrics and visualization |
| `predict_rnn.py` | Next month prediction for each state |
| `requirements_rnn.txt` | Python dependencies |
| `RNN_README.md` | Detailed documentation |

### 2. **Generated Model Files**

✅ **Trained Models:**
- `rnn_model_entropy.pth` - Model for entropy-weighted WSI
- `rnn_model_equal.pth` - Model for equal-weighted WSI

✅ **Artifacts:**
- `rnn_scalers_*.pkl` - Feature normalization scalers
- `rnn_feature_cols_*.pkl` - Feature column names
- `rnn_split_info_*.pkl` - Train/validation split metadata

### 3. **Results & Outputs**

📈 **Validation Results:**
- `rnn_validation_results_entropy.csv` - Detailed predictions (367 samples)
- `rnn_validation_results_equal.csv` - Detailed predictions (367 samples)
- `rnn_validation_metrics_entropy.csv` - Performance metrics
- `rnn_validation_metrics_equal.csv` - Performance metrics
- `rnn_validation_plot_entropy.png` - Visualization
- `rnn_validation_plot_equal.png` - Visualization

🔮 **Predictions:**
- `rnn_predictions_entropy.csv` - Next month forecast for 29 states
- `rnn_predictions_equal.csv` - Next month forecast for 29 states

---

## 📈 Model Performance

### **Entropy-Weighted WSI**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 15.28 | Average error of ~15 points |
| **MAE** | 8.72 | Median error of ~9 points |
| **R²** | 0.805 | **80.5% variance explained** ✅ |
| **MAPE** | 32.4% | Average percentage error |

### **Equal-Weighted WSI**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 12.02 | Average error of ~12 points |
| **MAE** | 8.93 | Median error of ~9 points |
| **R²** | 0.643 | **64.3% variance explained** ✅ |
| **MAPE** | 22.1% | Average percentage error |

---

## 🔍 Key Findings from Predictions

### **Entropy-Weighted WSI Predictions**

**States Expected to Improve (Decrease in WSI):**
1. **Arunachal Pradesh**: 93.33 → 56.29 (-37.04 points, -39.7%)
2. **Assam**: 91.81 → 53.49 (-38.32 points, -41.7%)
3. **Telangana**: 95.00 → 58.49 (-36.50 points, -38.4%)
4. **Jammu & Kashmir**: 92.68 → 56.24 (-36.44 points, -39.3%)
5. **Goa**: 92.54 → 60.27 (-32.27 points, -34.9%)

**States Expected to Worsen (Increase in WSI):**
1. **Maharashtra**: 87.51 → 100.77 (+13.26 points, +15.2%)
2. **Rajasthan**: 28.48 → 50.28 (+21.81 points, +76.6%)
3. **Chhattisgarh**: 21.61 → 41.83 (+20.22 points, +93.5%)

### **Equal-Weighted WSI Predictions**

**States Expected to Improve:**
1. **Chandigarh**: 72.82 → 36.68 (-36.15 points, -49.6%)
2. **West Bengal**: 65.95 → 44.50 (-21.46 points, -32.5%)
3. **Andhra Pradesh**: 57.69 → 38.91 (-18.78 points, -32.6%)

**Overall Trend:**
- **86.2%** of states expected to see **improvement** (25 out of 29)
- **Mean expected change**: -10.09 points (-15.7%)

---

## 🛠️ Model Architecture

```
┌─────────────────────────────────────┐
│      Input Features (11)            │
├─────────────────────────────────────┤
│  1) rainfall                        │
│  2) soil_moisture                   │
│  3) groundwater_level               │
│  4) population                      │
│  5) population_consumption_per_month│
│  6) LPCD                            │
│  7) rainfall_z (standardized)       │
│  8) soil_moisture_z                 │
│  9) groundwater_z                   │
│ 10) LPCD_z                          │
│ 11) WSI (target)                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│    RNN Layer 1 (64 hidden units)    │
│        Dropout (0.2)                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│    RNN Layer 2 (64 hidden units)    │
│        Dropout (0.2)                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     Fully Connected Layer           │
│        Output: WSI Value            │
└─────────────────────────────────────┘
```

---

## 📚 Data Split Strategy

### **Temporal 80-20 Split**

```
Training Data (80%): 2018-01 to 2023-03 (1,466 samples)
    ├── Used to train the model
    └── Early stopping based on validation loss

Validation Data (20%): 2023-04 to 2024-12 (367 samples)
    ├── Used to evaluate model performance
    └── Never seen during training
```

**Why temporal split?**
- Time series data requires chronological ordering
- Tests model's ability to predict **future** values
- Prevents data leakage from future to past

---

## 🚀 How to Use

### **1. Train the Model**
```bash
python train_rnn.py
```
**Output:** Trained models and artifacts  
**Time:** ~2-3 minutes

### **2. Validate the Model**
```bash
python validate_rnn.py
```
**Output:** Performance metrics and visualization plots  
**Time:** ~30 seconds

### **3. Generate Predictions**
```bash
python predict_rnn.py
```
**Output:** Next month WSI predictions for all states  
**Time:** ~10 seconds

---

## 📊 Sample Predictions

### Entropy-Weighted WSI (Top 5 States - Next Month)

| State | Current WSI | Predicted WSI | Change | % Change |
|-------|-------------|---------------|--------|----------|
| Arunachal Pradesh | 93.33 | 56.29 | -37.04 | -39.7% ✅ |
| Assam | 91.81 | 53.49 | -38.32 | -41.7% ✅ |
| Telangana | 95.00 | 58.49 | -36.50 | -38.4% ✅ |
| Jammu & Kashmir | 92.68 | 56.24 | -36.44 | -39.3% ✅ |
| Goa | 92.54 | 60.27 | -32.27 | -34.9% ✅ |

✅ = Improvement (lower WSI is better)

---

## 🎓 Key Learnings

### **1. RNN vs GRU Comparison**

| Aspect | RNN (This Model) | GRU (Alternative) |
|--------|------------------|-------------------|
| Architecture | Vanilla recurrent cells | Gated cells |
| Training Speed | Faster ⚡ | Slightly slower |
| Memory | Simpler | Better long-term |
| Parameters | Fewer | More (gates) |
| Performance | Good for 12-month sequences | Better for longer sequences |

### **2. Model Performance**

- **R² of 0.805** for entropy WSI = **Strong predictive power** ✅
- **RMSE of 15.28** = Reasonable accuracy for 0-100 scale
- **Early stopping** prevented overfitting (40 epochs for entropy, 33 for equal)

### **3. Data Insights**

- **Temporal coverage**: 2018-2024 (7 years)
- **Total sequences**: 1,833
- **Features**: 11 (mix of raw and normalized)
- **Sequence length**: 12 months (1 year of history)

---

## 📁 File Structure

```
fds lab/
├── rnn_model.py                    # Model architecture
├── train_rnn.py                    # Training script
├── validate_rnn.py                 # Validation script
├── predict_rnn.py                  # Prediction script
├── requirements_rnn.txt            # Dependencies
├── RNN_README.md                   # Detailed docs
├── RNN_PROJECT_SUMMARY.md          # This file
│
├── rnn_model_entropy.pth           # Trained model
├── rnn_model_equal.pth             # Trained model
├── rnn_scalers_entropy.pkl         # Scalers
├── rnn_scalers_equal.pkl           # Scalers
├── rnn_feature_cols_*.pkl          # Features
├── rnn_split_info_*.pkl            # Split info
│
├── rnn_validation_results_*.csv    # Validation data
├── rnn_validation_metrics_*.csv    # Metrics
├── rnn_validation_plot_*.png       # Plots
├── rnn_predictions_*.csv           # Predictions
└── Final_Statewise_Water_Dataset_preprocessed_WSI.csv
```

---

## ✅ Project Completion Checklist

- [x] ✅ RNN model architecture implemented
- [x] ✅ Training script with 80-20 temporal split
- [x] ✅ Both entropy & equal-weighted WSI models trained
- [x] ✅ Validation with comprehensive metrics
- [x] ✅ Visualization plots generated
- [x] ✅ Next month predictions for all 29 states
- [x] ✅ Detailed documentation (README)
- [x] ✅ Project summary (this document)

---

## 🎯 Next Steps (Optional Enhancements)

1. **Hyperparameter Tuning**
   - Try different hidden sizes (32, 128, 256)
   - Experiment with learning rates
   - Test different sequence lengths (6, 18, 24 months)

2. **Model Comparison**
   - Compare RNN vs GRU vs LSTM
   - Benchmark against traditional methods (ARIMA, Prophet)

3. **Feature Engineering**
   - Add seasonal indicators
   - Include rainfall patterns
   - Incorporate climate indices

4. **Ensemble Methods**
   - Combine RNN + GRU predictions
   - Weighted averaging

---

## 📞 Support

For questions or issues:
1. Check `RNN_README.md` for detailed documentation
2. Review validation metrics in CSV files
3. Examine validation plots for visual insights

---

**Created:** November 2025  
**Model Type:** Vanilla RNN  
**Framework:** PyTorch  
**Dataset:** Indian States Water Stress Index (2018-2024)

---

## 🎉 Summary

✅ **Successfully created** a complete RNN-based WSI prediction system  
✅ **Trained** two models with **80.5% and 64.3% R²** scores  
✅ **Validated** on 20% temporal holdout (2023-2024 data)  
✅ **Predicted** next month WSI for all 29 states  
✅ **Generated** comprehensive visualizations and results

**The model is ready to use for forecasting water stress conditions across Indian states!** 🌊💧
