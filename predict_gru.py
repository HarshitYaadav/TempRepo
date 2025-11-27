"""
Prediction script for GRU Water Stress Index Forecasting Model
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle
from gru_model import GRUForecaster, GRUTrainer
from sklearn.preprocessing import MinMaxScaler


def load_model_and_scalers(model_path, scalers_path, feature_cols_path, device='cpu'):
    """
    Load trained model, scalers, and feature columns
    
    Args:
        model_path: Path to saved model
        scalers_path: Path to saved scalers
        feature_cols_path: Path to saved feature columns
        device: Device to use
        
    Returns:
        model, scalers, feature_cols, model_config
    """
    # Load feature columns
    with open(feature_cols_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load scalers
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    # Load model checkpoint to get input size
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model (we need to infer input size from feature_cols)
    # We'll load the actual model state later
    model = GRUForecaster(
        input_size=len(feature_cols) + 1,  # +1 for target column
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model, scalers, feature_cols


def predict_future(
    df,
    model,
    scalers,
    feature_cols,
    target_column='WSI_entropy_0_100',
    sequence_length=12,
    forecast_months=12,
    train_split_ratio=0.8,
    device='cpu'
):
    """
    Predict future values for each state using ONLY training data
    
    Args:
        df: Historical DataFrame
        model: Trained GRU model
        scalers: Dictionary of scalers for each state
        feature_cols: List of feature column names
        target_column: Target column to predict
        sequence_length: Length of input sequence
        forecast_months: Number of months to forecast
        train_split_ratio: Ratio used for training (0.8 = use first 80% for predictions)
        device: Device to use
        
    Returns:
        DataFrame with predictions
    """
    predictions = []
    
    # Sort by state, year, month
    df = df.sort_values(['state', 'year', 'month']).reset_index(drop=True)
    
    print(f"\nIMPORTANT: Using ONLY training data (first {train_split_ratio*100:.0f}%) for predictions")
    print(f"This ensures predictions are made on unseen future data\n")
    
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        
        if state not in scalers:
            print(f"Warning: No scaler found for {state}, skipping...")
            continue
        
        # USE ONLY TRAINING DATA (first 80%) for predictions
        split_idx = int(len(state_df) * train_split_ratio)
        train_data = state_df.iloc[:split_idx].copy()
        
        if len(train_data) < sequence_length:
            print(f"Warning: Insufficient training data for {state} ({len(train_data)} samples), skipping...")
            continue
        
        scaler = scalers[state]
        
        # Get the last sequence_length months of TRAINING data only
        recent_data = train_data.tail(sequence_length)
        
        # Prepare features
        features = recent_data[feature_cols].values
        target = recent_data[target_column].values.reshape(-1, 1)
        
        # Combine for scaling
        combined = np.hstack([features, target])
        scaled_data = scaler.transform(combined)
        
        # Get the last sequence
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, -1)
        
        # Predict future months
        current_sequence = torch.FloatTensor(last_sequence).to(device)
        state_predictions = []
        
        # Get last known values
        last_year = recent_data['year'].iloc[-1]
        last_month = recent_data['month'].iloc[-1]
        
        # Keep track of the current sequence for rolling window
        current_scaled_data = scaled_data.copy()
        
        for month_ahead in range(1, forecast_months + 1):
            # Get current sequence (last sequence_length rows)
            current_sequence = current_scaled_data[-sequence_length:].reshape(1, sequence_length, -1)
            current_sequence_tensor = torch.FloatTensor(current_sequence).to(device)
            
            # Predict next value
            with torch.no_grad():
                pred = model(current_sequence_tensor)
                pred_value = pred.cpu().numpy()[0, 0]
            
            # Inverse transform the prediction
            # Create a dummy row with all features and the prediction
            # Use the last row's features (we'll use same features for simplicity)
            # In a real scenario, you might want to forecast features too
            last_row = current_scaled_data[-1].copy()
            last_row[-1] = pred_value  # Replace target with prediction
            
            # Inverse transform
            pred_unscaled = scaler.inverse_transform(last_row.reshape(1, -1))
            predicted_target = pred_unscaled[0, -1]
            
            state_predictions.append({
                'state': state,
                'year': last_year + (last_month + month_ahead - 1) // 12,
                'month': ((last_month + month_ahead - 1) % 12) + 1,
                'predicted_' + target_column: predicted_target,
                'month_ahead': month_ahead
            })
            
            # Update sequence for next prediction (using predicted value)
            # Append the new row with prediction to maintain rolling window
            current_scaled_data = np.vstack([current_scaled_data, last_row.reshape(1, -1)])
        
        predictions.extend(state_predictions)
    
    return pd.DataFrame(predictions)


def predict_single_index(df, index_name, target_column, script_path, 
                         SEQUENCE_LENGTH=12, FORECAST_MONTHS=12, TRAIN_SPLIT_RATIO=0.8, device='cpu'):
    """
    Generate predictions for a single WSI index using ONLY training data
    
    Args:
        df: Historical DataFrame
        index_name: Name of the index ('entropy' or 'equal')
        target_column: Target column name
        script_path: Path object for file paths
        TRAIN_SPLIT_RATIO: Ratio used for training (0.8 = use first 80% for predictions)
        Other args: Prediction parameters
    """
    print(f"\n{'='*60}")
    print(f"Generating predictions for {index_name.upper()} WSI index")
    print(f"Using ONLY training data (first {TRAIN_SPLIT_RATIO*100:.0f}%)")
    print(f"{'='*60}")
    
    # Paths with index name
    model_path = script_path.with_name(f"gru_model_{index_name}.pth")
    scalers_path = script_path.with_name(f"gru_scalers_{index_name}.pkl")
    feature_cols_path = script_path.with_name(f"gru_feature_cols_{index_name}.pkl")
    output_path = script_path.with_name(f"gru_predictions_{index_name}.csv")
    
    # Check if model exists
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print(f"Skipping {index_name} WSI predictions.")
        return None
    
    # Load model and scalers
    print("Loading trained model...")
    model, scalers, feature_cols = load_model_and_scalers(
        model_path, scalers_path, feature_cols_path, device
    )
    print(f"Model loaded. Features: {len(feature_cols)}")
    
    # Make predictions using ONLY training data
    print(f"\nGenerating predictions for next {FORECAST_MONTHS} months...")
    predictions_df = predict_future(
        df,
        model,
        scalers,
        feature_cols,
        target_column=target_column,
        sequence_length=SEQUENCE_LENGTH,
        forecast_months=FORECAST_MONTHS,
        train_split_ratio=TRAIN_SPLIT_RATIO,
        device=device
    )
    
    # Save predictions
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"States predicted: {predictions_df['state'].nunique()}")
    
    return predictions_df


def main():
    """Main prediction function - generates predictions for both WSI indices"""
    # Paths
    script_path = Path(__file__).resolve()
    data_path = script_path.with_name("Final_Statewise_Water_Dataset_preprocessed_WSI.csv")
    
    # Parameters
    SEQUENCE_LENGTH = 12
    FORECAST_MONTHS = 12  # Predict next 12 months
    TRAIN_SPLIT_RATIO = 0.8  # Use only first 80% (training data) for predictions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nIMPORTANT: Predictions will use ONLY training data (first {TRAIN_SPLIT_RATIO*100:.0f}%)")
    print(f"This ensures predictions are for truly future/unseen data\n")
    
    # WSI indices to predict
    wsi_indices = {
        'entropy': 'WSI_entropy_0_100',
        'equal': 'WSI_equal_0_100'
    }
    
    # Load data
    print("Loading historical data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    # Generate predictions for both indices
    all_predictions = {}
    for index_name, target_column in wsi_indices.items():
        pred_df = predict_single_index(
            df, index_name, target_column, script_path,
            SEQUENCE_LENGTH, FORECAST_MONTHS, TRAIN_SPLIT_RATIO, device
        )
        if pred_df is not None:
            all_predictions[index_name] = pred_df
    
    # Display summary
    if all_predictions:
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        for index_name, pred_df in all_predictions.items():
            print(f"\n{index_name.upper()} WSI:")
            print(f"  Total predictions: {len(pred_df)}")
            print(f"  States: {pred_df['state'].nunique()}")
            print(f"  Sample predictions:")
            print(pred_df.head(5).to_string(index=False))
        print("="*60)


if __name__ == "__main__":
    main()

