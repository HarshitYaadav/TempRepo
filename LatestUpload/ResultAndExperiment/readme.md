# EXPERIMENTATION AND RESULTS

## 5.1 Overview

This section presents a detailed overview of the experiments conducted to evaluate the performance of **AquaAlert's multi-index, multi-model forecasting framework**. The system was designed to predict monthly water stress conditions across Indian states using sophisticated machine learning approaches coupled with diverse index formulations.

The experimental framework encompasses:
- **Four Water Stress Index (WSI) formulations**: Equal-Weighted WSI, Entropy-Weighted WSI, PCA-Based WSI, and Hybrid (SPEI-Style Climatic) WSI
- **Three recurrent sequence models**: Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU)
- **Multi-state forecasting**: Predictions generated for multiple Indian states
- **Standardized evaluation metrics**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R² Score, and Mean Absolute Percentage Error (MAPE)

Through a structured experimentation pipeline, we aimed to identify the most accurate model–index combination for predicting next-month water stress, thereby enabling proactive water resource management and drought mitigation strategies.

---

## 5.2 Experimental Setup

### 5.2.1 Dataset and Preprocessing

**Data Source**: The experiments utilized the comprehensive `Final_Statewise_Water_Dataset_preprocessed_WSI_v3.csv` dataset, which contains:
- Monthly water stress indicators for multiple Indian states
- Temporal coverage spanning 2018-2020
- Multiple preprocessed WSI formulations with corresponding features

**Train-Validation Split**: 
- **Training Set**: 80% of temporal data (chronologically ordered)
- **Validation Set**: 20% of most recent temporal data
- **Splitting Strategy**: Time-series aware splitting to prevent data leakage
- **Normalization**: MinMaxScaler applied to features and target variables independently for each WSI formulation

### 5.2.2 Water Stress Index Formulations

Four distinct WSI calculation methodologies were implemented and evaluated:

1. **Equal-Weighted WSI**
   - Assigns uniform importance to all water stress indicators
   - Formula: Simple arithmetic mean of normalized features
   - Use case: Baseline model with no statistical bias

2. **Entropy-Weighted WSI**
   - Weights determined by information entropy of each indicator
   - Higher entropy features receive greater weight
   - Use case: Data-driven feature importance

3. **PCA-Based WSI**
   - Principal Component Analysis for dimensionality reduction
   - Captures maximum variance in first principal component
   - Use case: Identifies latent structure in water stress data

4. **Hybrid (SPEI-Style Climatic) WSI**
   - Integrates SPEI (Standardized Precipitation-Evapotranspiration Index)
   - Combines meteorological drought indicators with resource availability
   - Use case: Climate-aware water stress assessment

### 5.2.3 Model Architectures

All three recurrent neural network architectures were configured with identical hyperparameters for fair comparison:

| **Hyperparameter**         | **Value**              |
|----------------------------|------------------------|
| Input Sequence Length      | 3 months               |
| Hidden Layer Size          | 128 units              |
| Number of Hidden Layers    | 2                      |
| Dropout Rate               | 0.2                    |
| Learning Rate              | 0.001                  |
| Optimizer                  | Adam                   |
| Loss Function              | Mean Squared Error     |
| Batch Size                 | 32                     |
| Training Epochs            | 100                    |
| Early Stopping Patience    | 15 epochs              |

**Model Descriptions**:
- **RNN**: Vanilla recurrent neural network with basic temporal dependencies
- **LSTM**: Long Short-Term Memory networks with gating mechanisms to capture long-term dependencies
- **GRU**: Gated Recurrent Units with simplified gating structure compared to LSTM

### 5.2.4 Evaluation Metrics

Performance was assessed using four standard regression metrics:

1. **Root Mean Squared Error (RMSE)**
   ```
   RMSE = √(Σ(y_pred - y_actual)² / n)
   ```
   Lower values indicate better performance; penalizes large errors

2. **Mean Absolute Error (MAE)**
   ```
   MAE = Σ|y_pred - y_actual| / n
   ```
   Robust to outliers; interpretable in original units

3. **R² Score (Coefficient of Determination)**
   ```
   R² = 1 - (SS_res / SS_tot)
   ```
   Measures proportion of variance explained; ranges from -∞ to 1

4. **Mean Absolute Percentage Error (MAPE)**
   ```
   MAPE = (100/n) × Σ|(y_actual - y_pred) / y_actual|
   ```
   Scale-independent metric; expressed as percentage

> **Note on Metric Selection**: The combination of RMSE (penalizing large errors) and MAPE (percentage error) provides a balanced view of model performance, ensuring that predictions are accurate both in absolute terms and relative to the scale of water stress.

---

## 5.3 Experimental Results

### 5.3.1 Overall Performance Summary

> **Note**: The results highlight a fundamental trade-off: **PCA-based indices** offer superior variance explanation (high R²) suitable for research, while **Equal-Weighted indices** provide better absolute prediction accuracy (low RMSE) critical for operational use.

A total of **12 model-index combinations** were trained and evaluated (3 models × 4 indices). The table below summarizes the comprehensive performance metrics across all configurations:

| **Model** | **WSI Index**        | **RMSE ↓** | **MAE ↓** | **R² ↑** | **MAPE ↓** |
|-----------|----------------------|------------|-----------|----------|------------|
| **RNN**   | Equal-Weighted       | 12.049     | 8.973     | 0.641    | 22.396     |
| **RNN**   | Entropy-Weighted     | 14.951     | 8.093     | 0.814    | 33.008     |
| **RNN**   | PCA-Based            | **17.780** | 7.772     | **0.849**| 155.905    |
| **RNN**   | Hybrid (SPEI-Style)  | 12.382     | 9.056     | 0.643    | 31.642     |
| **LSTM**  | Equal-Weighted       | 11.829     | 8.988     | 0.654    | 21.790     |
| **LSTM**  | Entropy-Weighted     | 15.541     | 8.547     | 0.799    | 34.882     |
| **LSTM**  | PCA-Based            | **19.422** | 8.818     | 0.820    | **195.357**|
| **LSTM**  | Hybrid (SPEI-Style)  | 11.882     | 8.626     | 0.672    | 29.362     |
| **GRU**   | Equal-Weighted       | **11.782** | 8.889     | 0.657    | **21.245** |
| **GRU**   | Entropy-Weighted     | 15.230     | 8.271     | 0.807    | 30.984     |
| **GRU**   | PCA-Based            | 18.506     | 8.230     | 0.836    | 169.130    |
| **GRU**   | Hybrid (SPEI-Style)  | 11.787     | 8.620     | 0.677    | 29.462     |

**Key Observations**:
- **Best Overall Performance**: GRU with Equal-Weighted WSI (RMSE: 11.782, MAPE: 21.245%)
- **Highest R² Score**: RNN with PCA-Based WSI (R²: 0.849)
- **Lowest MAE**: RNN with PCA-Based WSI (MAE: 7.772)
- **Most Problematic**: PCA-Based WSI consistently produced high MAPE values across all models

### 5.3.2 Comparative Performance Analysis

#### **Comprehensive Performance Metrics**

![Overall Performance Matrix](file:///h:/2025%20winter/fds%20lab/results_plots/5_overall_performance_matrix.png)

**Consolidated Analysis**:
The 2×2 performance grid above summarizes the trade-offs between different model-index configurations:

*   **Prediction Accuracy (RMSE & MAE)**: The **GRU model with Equal-Weighted WSI** achieves the best operational performance (lowest RMSE: 11.782, MAE: 8.889), closely followed by LSTM. This configuration offers the most reliable absolute predictions.
*   **Variance Explanation (R²)**: **PCA-Based indices** consistently achieve the highest R² scores (>0.80) across all models, making them superior for research applications where explaining variability is prioritized over minimizing absolute error.
*   **Percentage Error (MAPE)**: A significant anomaly exists with PCA-based indices, where MAPE values spike (>150%) due to scale distortions from standardization. For percentage-based evaluation, Equal-Weighted and Hybrid indices are far superior (~21-29%).

**Key Takeaway**: There is a clear dichotomy between **accuracy-focused configurations** (Equal/Hybrid WSI) for deployment and **variance-focused configurations** (PCA/Entropy WSI) for research.

### 5.3.3 Model-Specific Performance

> **Note**: While LSTM models are often the default choice for time-series forecasting, our results indicate that the **GRU architecture**—with fewer parameters and simpler gating—achieves superior generalization on this hydrological dataset, suggesting that model complexity does not always equate to better performance.

#### **RNN Performance**
- **Strengths**: Achieves highest R² with PCA-Based WSI (0.849); lowest MAE with PCA (7.772)
- **Weaknesses**: Higher RMSE values compared to LSTM/GRU for most indices
- **Best Configuration**: PCA-Based WSI for variance explanation; Equal-Weighted WSI for balanced performance

#### **LSTM Performance**
- **Strengths**: Consistent mid-range performance across all indices; balanced RMSE and MAE
- **Weaknesses**: Highest RMSE with PCA-Based WSI (19.422); highest MAPE overall (195.357 with PCA)
- **Best Configuration**: Equal-Weighted WSI (RMSE: 11.829, MAPE: 21.790%)

#### **GRU Performance**
- **Strengths**: **Overall winner** with lowest RMSE and MAPE for Equal-Weighted WSI
- **Weaknesses**: No significant weaknesses; consistently competitive across all indices
- **Best Configuration**: Equal-Weighted WSI (RMSE: 11.782, MAE: 8.889, MAPE: 21.245%)

### 5.3.4 Index-Specific Performance

#### **Equal-Weighted WSI**
- **Best Model**: GRU (RMSE: 11.782, R²: 0.657)
- **Characteristics**: Most reliable and stable predictions; lowest MAPE values
- **Recommendation**: Preferred for operational deployment due to interpretability and consistent accuracy

#### **Entropy-Weighted WSI**
- **Best Model**: RNN (RMSE: 14.951, R²: 0.814)
- **Characteristics**: High R² scores across all models; moderate RMSE
- **Recommendation**: Suitable for exploratory analysis and variance explanation

#### **PCA-Based WSI**
- **Best Model**: RNN (RMSE: 17.780, R²: 0.849, MAE: 7.772)
- **Characteristics**: Highest R² scores but also highest MAPE values
- **Recommendation**: Use with caution; suitable for dimensionality reduction studies but problematic MAPE suggests percentage-based errors

#### **Hybrid (SPEI-Style) WSI**
- **Best Model**: GRU (RMSE: 11.787, R²: 0.677)
- **Characteristics**: Balanced performance similar to Equal-Weighted WSI
- **Recommendation**: Preferred for climate-informed forecasting applications

---

## 5.4 State-Level Forecasting Results

### 5.4.1 State Performance Extremes

The table below summarizes the best and worst performing states, highlighting regional disparities:

| **Region** | **Best Performing State** | **Worst Performing State** | **Primary Challenge** |
|------------|---------------------------|----------------------------|-----------------------|
| **Central** | **Chhattisgarh** (R²=0.845) | - | None (Best Region) |
| **East** | **Arunachal Pradesh** (R²=0.849) | West Bengal (R²=0.603) | Flood extremes |
| **North** | Uttar Pradesh (R²=0.802) | **Punjab** (R²=-0.244) | Groundwater depletion |
| **South** | Tamil Nadu (R²=0.771) | Kerala (High Bias) | Coastal complexity |
| **West** | Gujarat (R²=0.265) | **Rajasthan** (R²=0.185) | Aridity & Variability |

**Key Insights**:
*   **Top Performers**: Central and Eastern states (Arunachal, Chhattisgarh) show excellent predictability due to consistent hydrological patterns.
*   **Critical Failures**: Punjab and Rajasthan show poor or negative R² scores, indicating that the current model struggles with groundwater-dominated or highly arid regimes.
*   **Urban Success**: Delhi (High Stress) is predicted with remarkable accuracy (Error: 3.03), validating the model for urban water management.

### 5.4.3 Regional Performance Analysis

#### Regional Metrics Summary

![Regional Performance Comparison](file:///h:/2025%20winter/fds%20lab/results_plots/10_regional_performance_analysis.png)

| **Region** | **Avg RMSE** | **Avg MAE** | **Avg R²** | **States** | **Characteristics** |
|------------|--------------|-------------|------------|------------|---------------------|
| **Central** | 8.93 | 7.23 | **0.812** | 3 | Best performance; stable monsoon patterns |
| **East** | 11.48 | 9.23 | 0.684 | 9 | Good performance; high rainfall |
| **South** | 11.72 | 9.40 | 0.646 | 7 | Moderate; coastal-inland diversity |
| **North** | 13.46 | 10.42 | 0.320 | 7 | Challenging; urban stress, snow-fed rivers |
| **West** | **17.78** | **13.71** | **0.150** | 3 | Most challenging; arid climate |


| **Use Case**                          | **Recommended Configuration**      | **Justification**                                      |
|---------------------------------------|-------------------------------------|--------------------------------------------------------|
| **Operational Deployment**            | GRU + Equal-Weighted WSI           | Lowest RMSE (11.782) and MAPE (21.245%); interpretable |
| **Climate-Informed Forecasting**      | GRU + Hybrid (SPEI-Style) WSI      | Integrates meteorological drought; balanced metrics    |
| **Research & Variance Analysis**      | RNN + PCA-Based WSI                | Highest R² (0.849); best variance explanation          |
| **Data-Driven Feature Weighting**     | GRU + Entropy-Weighted WSI         | Strong R² (0.807); information theory-based weights    |

### 5.5.2 Key Findings

1. **GRU Superiority**: GRU architecture demonstrates the best overall performance, combining computational efficiency with prediction accuracy

2. **PCA Paradox**: PCA-Based WSI achieves highest R² scores (0.820-0.849) but suffers from extremely high MAPE values (155-195%), suggesting issues with percentage-based error calculations possibly due to standardization effects

3. **Equal-Weighted Reliability**: Equal-Weighted WSI provides the most balanced and stable predictions across all metrics

4. **Model Complexity Trade-off**: Despite LSTM's theoretical advantage in capturing long-term dependencies, GRU's simpler architecture yields comparable or better results with lower computational overhead

5. **Index Formulation Impact**: Choice of WSI formulation has greater impact on prediction accuracy than model architecture selection

### 5.5.3 Limitations and Future Work

**Current Limitations**:
- Limited temporal coverage (2018-2020) restricts long-term trend analysis
- MAPE metric unreliability for PCA-Based WSI requires alternative evaluation approaches
- State-wise performance variability not fully explored in aggregated metrics

**Future Research Directions**:
- **Ensemble Methods**: Combine predictions from multiple model-index pairs for improved robustness
- **Attention Mechanisms**: Integrate attention layers to identify critical temporal features
- **Spatial Dependencies**: Incorporate geographic proximity and inter-state water resource sharing
- **Extended Features**: Include socio-economic indicators (population growth, agricultural demand)
- **Multi-Step Forecasting**: Extend prediction horizon beyond one month
- **Uncertainty Quantification**: Implement probabilistic forecasting with confidence intervals

---



---

## 5.7 References and Supporting Materials

- **Complete Results Data**: `all_models_validation_summary.csv`
- **Model Checkpoints**: `{rnn|lstm|gru}_model_{equal|entropy|pca|hybrid}.pth`
- **Validation Results**: `{model}_validation_results_{index}.csv`
- **State-Level Predictions**: `{model}_predictions_{index}.csv`
- **Project Documentation**: `README.md`, `RNN_README.md`

---

**Document Version**: 1.0  
**Last Updated**: November 29, 2025  
**Framework**: AquaAlert Multi-Index WSI Forecasting System
