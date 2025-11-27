# RNN Model for Water Stress Index Prediction

This project implements a **Recurrent Neural Network (RNN)** model to predict Water Stress Index (WSI) for both entropy-weighted and equal-weighted indices.

## 📁 Project Structure

```
.
├── rnn_model.py               # RNN model architecture and trainer
├── train_rnn.py               # Training script
├── validate_rnn.py            # Validation script
├── predict_rnn.py             # Prediction script for next month
├── requirements_rnn.txt       # Dependencies
└── RNN_README.md              # This file
```

## 🎯 Features

- **RNN Architecture**: Vanilla RNN with multiple layers for time series forecasting
- **Dual Index Support**: Trains separate models for entropy and equal-weighted WSI
- **Temporal Split**: 80-20 split maintaining chronological order
- **Comprehensive Validation**: Multiple metrics (RMSE, MAE, R², MAPE)
- **Next Month Prediction**: Forecasts WSI for each state
- **Detailed Visualization**: Training curves and validation plots

## 📊 Model Architecture

```
Input Features (11):
├── rainfall
├── soil_moisture
├── groundwater_level
├── population
├── population_consumption_per_month
├── LPCD
├── rainfall_z (standardized)
├── soil_moisture_z (standardized)
├── groundwater_z (standardized)
├── LPCD_z (standardized)
└── WSI (target)

RNN Model:
├── Input Layer: 11 features
├── RNN Layers: 2 layers × 64 hidden units
├── Dropout: 0.2
├── Output Layer: 1 (predicted WSI)
└── Activation: tanh (RNN), linear (output)
```

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sequence Length | 12 | Months of history to use |
| Hidden Size | 64 | RNN hidden units |
| Num Layers | 2 | Number of RNN layers |
| Dropout | 0.2 | Dropout rate |
| Batch Size | 32 | Training batch size |
| Learning Rate | 0.001 | Adam optimizer LR |
| Max Epochs | 100 | Maximum training epochs |
| Train Split | 80% | Training data ratio |
| Val Split | 20% | Validation data ratio |

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements_rnn.txt
```

### 2. Train the Model

```bash
python train_rnn.py
```

**Outputs:**
- `rnn_model_entropy.pth` - Trained model for entropy WSI
- `rnn_model_equal.pth` - Trained model for equal WSI
- `rnn_scalers_*.pkl` - Feature scalers
- `rnn_feature_cols_*.pkl` - Feature column names
- `rnn_split_info_*.pkl` - Data split information

### 3. Validate the Model

```bash
python validate_rnn.py
```

**Outputs:**
- `rnn_validation_results_*.csv` - Detailed predictions vs actuals
- `rnn_validation_metrics_*.csv` - Performance metrics
- `rnn_validation_plot_*.png` - Visualization plots

### 4. Generate Predictions

```bash
python predict_rnn.py
```

**Outputs:**
- `rnn_predictions_*.csv` - Next month predictions for each state

## 📈 Data Preparation

### Temporal Split (80-20)

The dataset is split chronologically:
- **Training**: First 80% of time-ordered data
- **Validation**: Last 20% of time-ordered data

This ensures the model is evaluated on future (unseen) time periods.

### Sequence Creation

For each prediction:
```
Input: [t-11, t-10, ..., t-1, t] (12 months)
Output: [t+1] (next month)
```

### Feature Scaling

All features are normalized using `MinMaxScaler(0, 1)`:
- Fitted on training data only
- Applied to both training and validation sets
- Preserved for prediction phase

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **R²** | Coefficient of Determination |
| **MAPE** | Mean Absolute Percentage Error |

## 🎯 Model Differences: RNN vs GRU

| Aspect | RNN (This Model) | GRU (Alternative) |
|--------|------------------|-------------------|
| **Architecture** | Basic recurrent cells | Gated recurrent cells |
| **Parameters** | Fewer | More (gates) |
| **Training** | Faster | Slightly slower |
| **Memory** | Simpler | Better long-term dependencies |
| **Vanishing Gradient** | More prone | Less prone |

## 📝 Example Output

### Training
```
#############################################################
# Training RNN Model for WSI ENTROPY
#############################################################

Preparing data for WSI_entropy_0_100
============================================================
Features used (11):
  - rainfall
  - soil_moisture
  ...

Epoch [10/100], Train Loss: 0.012345, Val Loss: 0.013456
Epoch [20/100], Train Loss: 0.008765, Val Loss: 0.009876
...
```

### Validation
```
==================================================
VALIDATION METRICS (Original Scale)
==================================================
  MSE:  12.3456
  RMSE: 3.5136
  MAE:  2.7891
  R²:   0.8234
  MAPE: 4.56%
==================================================
```

### Prediction
```
Generating predictions...
  ✓ Andhra Pradesh          | Current:  75.32 | Predicted:  73.45 | Change:  -1.87
  ✓ Arunachal Pradesh       | Current:  68.21 | Predicted:  69.54 | Change:  +1.33
  ...
```

## 🔍 Validation Analysis

The validation script provides:

1. **Time Series Plot**: Shows actual vs predicted values over time
2. **Scatter Plot**: Shows correlation between actual and predicted
3. **Detailed CSV**: Contains per-sample predictions and errors
4. **Summary Statistics**: Overall performance metrics

## 🌟 Key Features

1. **Temporal Consistency**: Maintains chronological order in data split
2. **State-wise Predictions**: Generates forecasts for each state individually
3. **Dual Index Support**: Handles both WSI calculation methods
4. **Early Stopping**: Prevents overfitting with patience mechanism
5. **Gradient Clipping**: Prevents exploding gradients
6. **Learning Rate Scheduling**: Adaptive learning rate adjustment

## ⚠️ Important Notes

1. **Data Requirements**: 
   - Preprocessed dataset `Final_Statewise_Water_Dataset_preprocessed_WSI.csv` must be present
   - Minimum 12 months of data required per state for predictions

2. **Model Files**:
   - Models are saved in PyTorch format (.pth)
   - Scalers and metadata saved as pickle files

3. **Device Support**:
   - Automatically uses CUDA if available
   - Falls back to CPU otherwise

## 📚 References

- Uses vanilla RNN architecture with tanh activation
- Implements temporal 80-20 split for time series validation
- Follows best practices for time series forecasting

## 🤝 Contributing

This is an academic project for FDS Lab. For questions or improvements, please contact the course instructor.

---

**Last Updated**: November 2025
