import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

def main():
    script_path = Path(__file__).resolve()
    wsi_path = script_path.with_name("Final_Statewise_Water_Dataset_preprocessed_WSI_v3.csv")
    spei_path = script_path.with_name("Statewise_SPEI_India_2018_2020.csv")

    if not wsi_path.exists():
        print(f"Error: WSI file not found at {wsi_path}")
        return
    if not spei_path.exists():
        print(f"Error: SPEI file not found at {spei_path}")
        return

    print("Loading datasets...")
    df_wsi = pd.read_csv(wsi_path)
    df_spei = pd.read_csv(spei_path)

    # Standardize state names if necessary (simple check)
    # df_wsi['state'] = df_wsi['state'].str.strip()
    # df_spei['state'] = df_spei['state'].str.strip()

    print("Merging datasets on Year, Month, State...")
    # Merge using inner join to keep only matching records
    merged = pd.merge(
        df_wsi, 
        df_spei, 
        on=["year", "month", "state"], 
        how="inner"
    )

    print(f"Merged dataset has {len(merged)} rows.")

    if len(merged) == 0:
        print("Warning: No matching records found. Check state names or date ranges.")
        return

    # Indices to correlate
    wsi_cols = ["WSI_equal", "WSI_entropy", "WSI_pca", "WSI_hybrid"]
    spei_col = "spei1" # Using 1-month SPEI as WSI is monthly

    with open("correlation_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Correlation Analysis (N={len(merged)}):\n")
        f.write(f"Comparing WSI indices (Higher=Stress) vs SPEI (Lower=Drought/Stress)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Index':<15} | {'Pearson r':<12} | {'P-value':<10} | {'Spearman rho':<12} | {'P-value':<10}\n")
        f.write("-" * 60 + "\n")

        results = {}

        for col in wsi_cols:
            if col not in merged.columns:
                f.write(f"Warning: {col} not found in merged dataset.\n")
                continue
            
            # Drop NaNs for correlation
            valid_data = merged[[col, spei_col]].dropna()
            
            if len(valid_data) < 2:
                f.write(f"Not enough data for {col}\n")
                continue

            x = valid_data[col]
            y = valid_data[spei_col]

            p_r, p_p = pearsonr(x, y)
            s_r, s_p = spearmanr(x, y)

            results[col] = {"pearson": p_r, "spearman": s_r}

            f.write(f"{col:<15} | {p_r:12.4f} | {p_p:10.4f} | {s_r:12.4f} | {s_p:10.4f}\n")

        f.write("-" * 60 + "\n")
        f.write("\nInterpretation:\n")
        f.write("Since Higher WSI = Higher Stress and Lower SPEI = Higher Stress (Drought),\n")
        f.write("we expect a NEGATIVE correlation.\n")
        
        # Check for strongest correlation
        best_idx = None
        best_corr = 0
        
        for col, res in results.items():
            # We look for the most negative correlation
            if res["pearson"] < best_corr:
                best_corr = res["pearson"]
                best_idx = col
                
        if best_idx:
            f.write(f"\nStrongest negative correlation with SPEI is: {best_idx} (r={best_corr:.4f})\n")
    
    print("Analysis complete. Results saved to correlation_summary.txt")

if __name__ == "__main__":
    main()
