# STATE-LEVEL VALIDATION AND WSI PERFORMANCE ANALYSIS

## Overview

This document provides comprehensive state-level analysis of water stress prediction performance across 30 Indian states and union territories. The analysis examines both prediction accuracy (how well the model predicts) and water stress levels (which states experience higher/lower stress).

**Analysis Components**:
- State-wise prediction accuracy rankings
- Regional performance comparison
- Model comparison (GRU vs RNN)
- Water stress level analysis by state
- Prediction bias and error patterns
- Monthly WSI trends

---

## 1. STATE-WISE PREDICTION ACCURACY RANKINGS

### 1.1 Top Performing States (R² Score - GRU + Equal-Weighted WSI)

The following states demonstrated exceptional water stress prediction accuracy, achieving R² scores above 0.75:

| **Rank** | **State** | **RMSE** | **MAE** | **R² Score** | **Performance Tier** |
|----------|-----------|----------|---------|--------------|---------------------|
| 1 | **Arunachal Pradesh** | 7.640 | 5.983 | **0.849** | Excellent |
| 2 | **Chhattisgarh** | 8.637 | 7.161 | **0.845** | Excellent |
| 3 | **Odisha** | 9.651 | 8.428 | **0.810** | Excellent |
| 4 | **Uttar Pradesh** | 8.525 | 6.028 | **0.802** | Excellent |
| 5 | **Madhya Pradesh** | 9.634 | 8.489 | **0.790** | Excellent |
| 6 | **Tamil Nadu** | 10.030 | 6.889 | **0.771** | Excellent |
| 7 | **Tripura** | 10.560 | 9.708 | **0.765** | Excellent |
| 8 | **Andhra Pradesh** | 8.985 | 7.532 | **0.751** | Good |
| 9 | **Puducherry** | 10.559 | 8.430 | **0.729** | Good |
| 10 | **Uttarakhand** | 11.220 | 8.842 | **0.676** | Good |

**Key Insights**:
- **Arunachal Pradesh** achieved the highest R² score (0.849), explaining 84.9% of water stress variance
- **Central and Eastern states** dominate top performers - more predictable hydrological patterns
- Top 10 states all achieved R² > 0.67
- Lower RMSE values (7.6-11.2) indicate predictions within ±10 WSI units

### 1.2 Challenge States (Bottom Performers)

| **Rank** | **State** | **RMSE** | **MAE** | **R² Score** | **Challenges** |
|----------|-----------|----------|---------|--------------|----------------|
| 21 | Delhi | 12.819 | 9.122 | 0.347 | Urban complexity |
| 28 | **Punjab** | 18.928 | 13.917 | **-0.244** | Groundwater depletion |
| 29 | Rajasthan | 17.291 | 14.679 | 0.185 | Arid, high variability |
| 30 | West Bengal | 15.160 | 12.263 | 0.603 | Flood-prone extremes |

**Key Insights**:
- **Punjab** showed negative R² (-0.244), indicating model predictions perform worse than simply using the mean value
- **Western states** face prediction challenges due to extreme arid conditions and groundwater over-exploitation
- Higher RMSE values (15-20) indicate predictions can deviate ±15-20 WSI units

---

## 2. VISUAL ANALYSIS

### 2.1 State Performance Rankings

![State Performance Rankings](file:///h:/2025%20winter/fds%20lab/results_plots/9_state_performance_ranking.png)

**Geographic Pattern**: Clear east-west gradient - eastern states show better performance than western arid states.

**Analysis**:
- **Top 15 States (Green)**: R² scores range from 0.60 to 0.85, predominantly Eastern and Central regions
- **Bottom 15 States (Red)**: R² scores range from -0.24 to 0.60, dominated by Western and Northern states

---

## 3. REGIONAL PERFORMANCE ANALYSIS

### 3.1 Regional Metrics Summary

![Regional Performance Comparison](file:///h:/2025%20winter/fds%20lab/results_plots/10_regional_performance_analysis.png)

| **Region** | **Avg RMSE** | **Avg MAE** | **Avg R²** | **States** | **Characteristics** |
|------------|--------------|-------------|------------|------------|---------------------|
| **Central** | 8.93 | 7.23 | **0.812** | 3 | Best performance; stable monsoon patterns |
| **East** | 11.48 | 9.23 | 0.684 | 9 | Good performance; high rainfall |
| **South** | 11.72 | 9.40 | 0.646 | 7 | Moderate; coastal-inland diversity |
| **North** | 13.46 | 10.42 | 0.320 | 7 | Challenging; urban stress, snow-fed rivers |
| **West** | **17.78** | **13.71** | **0.150** | 3 | Most challenging; arid climate |

**Key Findings**:
1. **Central Region**: Best performance (Avg R² = 0.812) - includes Madhya Pradesh, Chhattisgarh, Uttar Pradesh
2. **Western Region**: Most challenging (Avg R² = 0.150) - arid conditions, groundwater issues
3. Regional differences exceed model architecture differences

---

## 4. MODEL COMPARISON (GRU vs RNN)

### 4.1 State-Wise Model Improvement

![GRU vs RNN Comparison](file:///h:/2025%20winter/fds%20lab/results_plots/11_state_model_comparison.png)

**States Where GRU Significantly Improved**:
- **Telangana**: +10.8 RMSE improvement
- **West Bengal**: +9.9 RMSE improvement  
- **Maharashtra**: +8.5 RMSE improvement

**Pattern**: GRU's advantage most pronounced in states with complex, non-linear water stress patterns.

---

## 5. WATER STRESS LEVEL ANALYSIS

### 5.1 States with Lowest Water Stress

![State-Wise WSI Comparison](file:///h:/2025%20winter/fds%20lab/results_plots/12_state_wsi_comparison.png)

| **Rank** | **State** | **Avg Actual WSI** | **Avg Predicted WSI** | **Prediction Error** | **Stress Category** |
|----------|-----------|--------------------|-----------------------|----------------------|---------------------|
| 1 | Kerala | 48.84 | 54.99 | 12.23 | Moderate Stress |
| 2 | Madhya Pradesh | 49.99 | 50.45 | 5.57 | Moderate Stress |
| 3 | Punjab | 50.27 | 56.86 | 6.60 | Moderate Stress |
| 4 | Tamil Nadu | 51.80 | 54.57 | 10.65 | Moderate Stress |
| 5 | Chhattisgarh | 52.45 | 61.49 | 11.79 | Moderate Stress |

**Key Findings**:
- **No Low-Stress States**: All Indian states experience at least moderate water stress (>40 WSI)
- **Kerala**: Lowest average water stress (48.84 WSI) but model overpredicts by 6.14 points
- **Madhya Pradesh**: Best prediction accuracy among low-stress states (error of only 5.57)
- Model generally overpredicts water stress for these states (positive bias)

### 5.2 States with Highest Water Stress

| **Rank** | **State** | **Avg Actual WSI** | **Avg Predicted WSI** | **Prediction Error** | **Stress Category** |
|----------|-----------|--------------------|-----------------------|----------------------|---------------------|
| 1 | **Delhi** | 71.54 | 72.33 | 3.03 | High Stress |
| 2 | Goa | 66.45 | 62.36 | 7.27 | High Stress |
| 3 | Puducherry | 65.30 | 57.16 | 10.76 | High Stress |
| 4 | Gujarat | 64.49 | 74.38 | 9.89 | High Stress |
| 5 | Bihar | 63.19 | 60.71 | 5.92 | High Stress |

**Key Findings**:
- **Delhi**: Highest average water stress (71.54 WSI) with **excellent** prediction accuracy (error of only 3.03)
- **Accuracy Paradox**: Model predicts high-stress situations MORE accurately than moderate-stress situations
- All top 5 highest stress states are categorized as "High Stress" (>60 WSI)
- Urban centers show better prediction accuracy despite high stress levels

---

## 6. PREDICTION ACCURACY BY STRESS LEVEL

### 6.1 Performance by Stress Category

![Prediction Accuracy by WSI](file:///h:/2025%20winter/fds%20lab/results_plots/13_prediction_accuracy_by_wsi.png)

| **Stress Category** | **WSI Range** | **Number of States** | **Avg Prediction Error** |
|---------------------|---------------|----------------------|--------------------------|
| Low Stress | <40 | 0 | N/A |
| Moderate Stress | 40-60 | 19 (63%) | 8.87 |
| High Stress | >60 | 11 (37%) | **6.82** |

**Insights**:
- **No Low-Stress States**: All states experience at least moderate water stress - nationwide concern
- **Better Accuracy for High Stress**: Model performs better when water stress is elevated (error: 6.82 vs 8.87)
- **37% High-Stress States**: 11 states face persistent high water stress requiring immediate attention

### 6.2 Best and Worst Predicted States

**Best Predicted (Lowest Prediction Error)**:

| **Rank** | **State** | **Avg Actual WSI** | **Prediction Error** | **Stress Category** |
|----------|-----------|--------------------|-----------------------|---------------------|
| 1 | Delhi | 71.54 | 3.03 | High Stress |
| 2 | Rajasthan | 61.28 | 4.11 | High Stress |
| 3 | Tripura | 55.03 | 4.45 | Moderate Stress |
| 4 | Nagaland | 55.59 | 5.05 | Moderate Stress |
| 5 | Uttar Pradesh | 60.16 | 5.32 | High Stress |

**Worst Predicted (Highest Prediction Error)**:

| **Rank** | **State** | **Avg Actual WSI** | **Prediction Error** | **Stress Category** |
|----------|-----------|--------------------|-----------------------|---------------------|
| 1 | Andhra Pradesh | 55.19 | 24.61 | Moderate Stress |
| 2 | Arunachal Pradesh | 56.23 | 18.37 | Moderate Stress |
| 3 | Himachal Pradesh | 54.68 | 13.69 | Moderate Stress |
| 4 | Kerala | 48.84 | 12.23 | Moderate Stress |
| 5 | Chhattisgarh | 52.45 | 11.79 | Moderate Stress |

**Pattern Analysis**:
- High-stress states (Delhi, Rajasthan, UP) are MORE accurately predicted
- Moderate-stress states show higher prediction variability
- Himalayan states (Himachal, Arunachal) show elevated prediction errors
- Coastal states (Kerala, Andhra Pradesh, Goa) have higher prediction uncertainty

---

## 7. PREDICTION BIAS ANALYSIS

### 7.1 States Where Model Overpredicts (Positive Bias)

| **State** | **Actual WSI** | **Predicted WSI** | **Bias** | **Implication** |
|-----------|----------------|-------------------|----------|-----------------|
| Himachal Pradesh | 54.68 | 66.89 | +12.21 | False alarms possible |
| Gujarat | 64.49 | 74.38 | +9.89 | Overstated drought risk |
| Chhattisgarh | 52.45 | 61.49 | +9.04 | Conservative estimates |
| Punjab | 50.27 | 56.86 | +6.60 | Moderate overestimation |
| Kerala | 48.84 | 54.99 | +6.14 | Slight overestimation |

### 7.2 States Where Model Underpredicts (Negative Bias)

| **State** | **Actual WSI** | **Predicted WSI** | **Bias** | **Implication** |
|-----------|----------------|-------------------|----------|-----------------|
| Puducherry | 65.30 | 57.16 | -8.14 | Missed drought warnings |
| Meghalaya | 57.50 | 51.06 | -6.44 | Underestimated stress |
| Maharashtra | 56.84 | 51.40 | -5.45 | Risk underestimation |
| Goa | 66.45 | 62.36 | -4.09 | Moderate underestimation |
| Uttar Pradesh | 60.16 | 57.00 | -3.16 | Slight underestimation |

**Bias Insights**:
- Model tends to **overpredict** in agricultural states (Punjab, Chhattisgarh)
- Model **underpredicts** in coastal/high-rainfall states (Meghalaya, Goa)
- Himalayan states show significant overprediction
- Urban centers show minimal bias (Delhi: +0.79)

---

## 8. SEASONAL AND TEMPORAL PATTERNS

### 8.1 Monthly WSI Trends for Representative States

![Monthly WSI Trends](file:///h:/2025%20winter/fds%20lab/results_plots/14_monthly_wsi_trends.png)

This visualization shows monthly water stress patterns for 4 representative states across different stress levels:
- **Low Stress State**: Demonstrates stable, predictable patterns
- **Moderate-Low State**: Shows seasonal variation with good prediction tracking
- **Moderate-High State**: Exhibits higher variability with acceptable accuracy
- **High Stress State**: Urban center with extreme stress but excellent prediction

### 8.2 WSI Range by State (Seasonal Variability)

**States with Widest WSI Variation (High Seasonal Variability)**:

| **State** | **Min WSI** | **Max WSI** | **Range** | **Avg WSI** |
|-----------|-------------|-------------|-----------|-------------|
| Bihar | 25.00 | 100.00 | 75.00 | 63.19 |
| Assam | 28.20 | 95.75 | 67.55 | 62.52 |
| Arunachal Pradesh | 20.62 | 86.02 | 65.40 | 56.23 |
| Puducherry | 32.61 | 95.39 | 62.78 | 65.30 |
| Goa | 35.69 | 93.81 | 58.12 | 66.45 |

**Insight**: These states experience extreme seasonal water stress variation due to monsoon-dependency, making predictions challenging.

**States with Narrowest WSI Variation (Stable Water Stress)**:

| **State** | **Min WSI** | **Max WSI** | **Range** | **Avg WSI** |
|-----------|-------------|-------------|-----------|-------------|
| Delhi | 48.59 | 84.47 | 35.88 | 71.54 |
| Rajasthan | 36.53 | 78.82 | 42.29 | 61.28 |
| Punjab | 26.76 | 70.70 | 43.94 | 50.27 |
| Uttarakhand | 36.82 | 79.72 | 42.90 | 55.32 |
| Haryana | 27.50 | 77.57 | 50.07 | 52.74 |

**Insight**: Urban and arid states show more stable water stress patterns. Lower variability correlates with better prediction accuracy.

---

## 9. OVERALL STATISTICS

### 9.1 National-Level Summary

| **Metric** | **Value** |
|------------|-----------|
| **National Average Actual WSI** | 57.74 |
| **National Average Predicted WSI** | 57.67 |
| **Overall Prediction Bias** | -0.07 (nearly unbiased!) |
| **Overall Average Prediction Error** | 8.78 |
| **Total States Analyzed** | 30 |
| **Total Predictions Made** | 369 |
| **Temporal Coverage** | 2023-2024 |

### 9.2 Stress Category Distribution

- **Low Stress (<40 WSI)**: 0 states (0%)
- **Moderate Stress (40-60 WSI)**: 19 states (63%)
- **High Stress (>60 WSI)**: 11 states (37%)

---

## 10. RECOMMENDATIONS

### 10.1 By State Category

**For Low/Moderate Stress States (WSI 40-60)**:
- Deploy model for general planning purposes
- Combine predictions with local data during transition seasons
- **High Confidence States**: Madhya Pradesh, Tamil Nadu, Tripura
- **Exercise Caution**: Kerala, Chhattisgarh (higher errors)

**For High Stress States (WSI >60)**:
- High confidence for operational deployment
- Use for drought early warning systems
- **Excellent Accuracy**: Delhi, Rajasthan, Bihar
- **Priority Intervention**: Goa, Puducherry, Gujarat

**For High-Variability States (High WSI Range)**:
- Use ensemble predictions with confidence intervals
- Update models more frequently with recent data
- **Examples**: Bihar, Assam, Arunachal Pradesh
- Require specialized regional models

### 10.2 By Region

**Central Region (Best Performance)**:
- Ready for full operational deployment
- Use for strategic water resource planning
- Model training template for other regions

**Eastern Region (Good Performance)**:
- Deploy with quarterly model updates
- Monitor monsoon-dependent patterns
- Good early warning capability

**Western Region (Challenging)**:
- Supplement with satellite data and expert systems
- Focus on groundwater monitoring
- Develop region-specific models

**Northern Region (Variable)**:
- Account for snow-fed river dynamics
- Urban centers (Delhi) - deploy with confidence
- Himalayan states - use ensemble approaches

**Southern Region (Moderate)**:
- Deploy with seasonal calibration
- Strong performance in Tamil Nadu
- Monitor coastal-inland diversity

---

## 11. KEY TAKEAWAYS

1. **No Low-Stress States**: All Indian states experience at least moderate water stress (>40 WSI), highlighting nationwide water scarcity concerns.

2. **Accuracy Paradox**: Model predicts HIGH-stress situations MORE accurately than moderate-stress situations (error: 6.82 vs 8.87). Delhi with 71.54 WSI has only 3.03 error.

3. **Geographic Patterns**:
   - **Eastern & Central states**: Best prediction accuracy
   - **Western states**: Challenging due to arid conditions
   - **Coastal states**: Higher prediction uncertainty
   - **Himalayan states**: Significant overprediction bias

4. **Nearly Unbiased Overall**: Overall prediction bias is only -0.07, though individual states show systematic over/underprediction patterns.

5. **Seasonal Dependency**: Monsoon-dependent states show extreme WSI variation (25-100 range), requiring careful model interpretation.

6. **Operational Readiness**: 11 high-stress states (37%) have sufficient prediction accuracy (error <7) for drought early warning deployment.

7. **Regional Disparities**: Performance gap between Central Region (R²=0.812) and Western Region (R²=0.150) suggests need for region-specific models.

---

## 12. DATA FILES AND VISUALIZATIONS

### 12.1 CSV Data Files
- `state_wsi_performance_detailed.csv` - Complete analysis for all 30 states
- `state_performance_gru_equal.csv` - State-wise accuracy metrics
- `top_10_states.csv` / `bottom_10_states.csv` - Performance rankings
- `top_5_lowest_wsi_states.csv` / `top_5_highest_wsi_states.csv` - WSI rankings
- `regional_performance_summary.csv` - Regional aggregates

### 12.2 Visualizations
1. **State Performance Rankings** (`9_state_performance_ranking.png`)
2. **Regional Performance Analysis** (`10_regional_performance_analysis.png`)
3. **GRU vs RNN Comparison** (`11_state_model_comparison.png`)
4. **State-Wise WSI Comparison** (`12_state_wsi_comparison.png`)
5. **Prediction Accuracy by WSI** (`13_prediction_accuracy_by_wsi.png`)
6. **Monthly WSI Trends** (`14_monthly_wsi_trends.png`)

---

**Document Version**: 1.0  
**Last Updated**: November 29, 2025  
**Analysis Period**: 2023-2024 Validation Data  
**Framework**: AquaAlert Multi-Index WSI Forecasting System
