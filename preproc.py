from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_dataset_paths() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    input_csv = script_path.with_name("Final_Statewise_Water_Dataset.csv")
    output_csv = script_path.with_name("Final_Statewise_Water_Dataset_preprocessed_WSI.csv")
    return input_csv, output_csv


def interpolate_and_fill_by_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_interp = ["rainfall", "soil_moisture", "groundwater_level"]
    for col in cols_to_interp:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interpolate within each state to preserve trends
    df[cols_to_interp] = (
        df.groupby("state", group_keys=False)[cols_to_interp]
        .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
    )

    # Fill any remaining gaps with state means
    df[cols_to_interp] = (
        df.groupby("state", group_keys=False)[cols_to_interp].apply(lambda g: g.fillna(g.mean()))
    )
    return df


def add_lpcd_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["population_consumption_per_month"] = pd.to_numeric(
        df["population_consumption_per_month"], errors="coerce"
    )

    denom = df["population"] * 30
    lpcd = df["population_consumption_per_month"] / denom
    lpcd = lpcd.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], pd.NA)
    df["LPCD"] = lpcd
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    filtered = df.copy()
    for col in cols:
        series = pd.to_numeric(filtered[col], errors="coerce")
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filtered = filtered[(series >= lower) & (series <= upper)]
    return filtered


def add_zscores(df: pd.DataFrame, cols: list[str], out_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[out_cols] = scaler.fit_transform(df[cols])
    return df


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)


def compute_entropy_weights(X: pd.DataFrame) -> np.ndarray:
    X_safe = X.fillna(0.0).clip(lower=0.0)
    col_sums = X_safe.sum(axis=0).replace(0, np.nan)
    P = X_safe.divide(col_sums, axis=1)
    n = len(X_safe)
    uniform = np.full((n,), 1.0 / n)
    for j, col in enumerate(X_safe.columns):
        if np.isnan(col_sums[col]):
            P[col] = uniform
    k = 1.0 / np.log(n) if n > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        P_vals = P.values
        logP = np.where(P_vals > 0, np.log(P_vals), 0.0)
        e = -k * np.sum(P_vals * logP, axis=0)
    d = 1.0 - e
    if np.allclose(d.sum(), 0.0) or np.isnan(d.sum()):
        w = np.full((X_safe.shape[1],), 1.0 / X_safe.shape[1])
    else:
        w = d / d.sum()
    return w


def add_wsi_entropy_and_equal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # Scarcity-oriented indicators in [0,1]
    lpcd_s = _minmax(df["LPCD"])             # demand (higher = more stress)
    rainfall_s = 1.0 - _minmax(df["rainfall"])       # higher rainfall = less stress
    soil_s = 1.0 - _minmax(df["soil_moisture"])      # higher soil = less stress
    gw_s = 1.0 - _minmax(df["groundwater_level"])    # higher gw = less stress

    X = pd.DataFrame({
        "LPCD_s": lpcd_s,
        "rainfall_s": rainfall_s,
        "soil_s": soil_s,
        "groundwater_s": gw_s,
    }, index=df.index)

    # Compute entropy weights
    weights = compute_entropy_weights(X)
    weight_map = {col: float(w) for col, w in zip(X.columns, weights)}

    # Entropy-weighted WSI
    df["WSI_entropy"] = X.values.dot(weights)
    df["WSI_entropy_0_100"] = _minmax(df["WSI_entropy"]) * 100.0

    # Equal-weight WSI
    df["WSI_equal"] = X.mean(axis=1)
    df["WSI_equal_0_100"] = _minmax(df["WSI_equal"]) * 100.0

    # Add weight columns for easy Excel viewing (same value repeated so visible in all rows)
    for k, v in weight_map.items():
        df[f"entropy_weight_{k}"] = v

    return df, weight_map


def main() -> None:
    input_csv, output_csv = get_dataset_paths()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load
    df = pd.read_csv(input_csv)

    # 1) Handle missing
    df = interpolate_and_fill_by_state(df)

    # 2) Add LPCD
    df = add_lpcd_column(df)

    # 3) Remove outliers
    cols_for_outliers = ["rainfall", "soil_moisture", "groundwater_level", "LPCD"]
    df = remove_outliers_iqr(df, cols_for_outliers)

    # 4) Add z-scores
    z_in = ["rainfall", "soil_moisture", "groundwater_level", "LPCD"]
    z_out = ["rainfall_z", "soil_moisture_z", "groundwater_z", "LPCD_z"]
    df = add_zscores(df, z_in, z_out)

    # 5) Compute both WSI versions
    df, weights = add_wsi_entropy_and_equal(df)

    # 6) Save
    df.to_csv(output_csv, index=False)

    print("Entropy Weights (scarcity indicators):")
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved output with both WSI columns and weights to: {output_csv}")


if __name__ == "__main__":
    main()
