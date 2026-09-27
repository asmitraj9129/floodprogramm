import pandas as pd
import numpy as np
from pathlib import Path
from data.validation import MissingValueHandler, OutlierDetector, SourceReliabilityEvaluator

class RainfallPipeline:
    """
    Ingests raw rainfall datasets (historical, observed, simulated), cleans anomalies,
    computes temporal accumulation windows (1h, 3h, 6h), and tags provenance metadata.
    """
    
    def __init__(self, historical_path: str = None, simulated_path: str = None):
        self.historical_path = historical_path or "data/raw/rainfall/rainfall_historical.csv"
        self.simulated_path = simulated_path or "data/raw/rainfall/rainfall_simulated.csv"

    def process(self) -> pd.DataFrame:
        dfs = []
        
        # Load historical rain data if exists
        if Path(self.historical_path).exists():
            df_hist = pd.read_csv(self.historical_path)
            if "data_type" not in df_hist.columns:
                df_hist["data_type"] = "historical"
            dfs.append(df_hist)
            
        # Load simulated real-time rain feed if exists
        if Path(self.simulated_path).exists():
            df_sim = pd.read_csv(self.simulated_path)
            if "data_type" not in df_sim.columns:
                df_sim["data_type"] = "simulated"
            dfs.append(df_sim)
            
        if not dfs:
            raise FileNotFoundError("No raw rainfall files found at configured paths.")
            
        raw_df = pd.concat(dfs, ignore_index=True)
        
        # 1. Standardize timestamp
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
        raw_df = raw_df.sort_values(by=["sub_basin_id", "timestamp"]).reset_index(drop=True)
        
        # 2. Outlier and Physical Limit Handling
        raw_df, _ = OutlierDetector.enforce_physical_bounds(raw_df)
        raw_df = OutlierDetector.cap_zscore_outliers(raw_df, cols=["rainfall_mm_1h"], threshold=4.5)
        
        # 3. Missing values handling
        raw_df = MissingValueHandler.impute_rainfall(raw_df, col="rainfall_mm_1h", group_col="sub_basin_id")
        
        # 4. Feature Engineering: Rolling accumulations per sub-basin
        processed_records = []
        for sub_id, group in raw_df.groupby("sub_basin_id"):
            group = group.copy().sort_values("timestamp")
            
            # Calculate rolling sums (assuming 1-hour resolution time steps)
            group["rainfall_mm_1h"] = group["rainfall_mm_1h"].round(2)
            group["rainfall_mm_3h"] = group["rainfall_mm_1h"].rolling(window=3, min_periods=1).sum().round(2)
            group["rainfall_mm_6h"] = group["rainfall_mm_1h"].rolling(window=6, min_periods=1).sum().round(2)
            group["rainfall_intensity_max"] = group["rainfall_mm_1h"].rolling(window=3, min_periods=1).max().round(2)
            
            processed_records.append(group)
            
        final_df = pd.concat(processed_records, ignore_index=True)
        
        # 5. Metadata & Reliability Tagging
        final_df = SourceReliabilityEvaluator.assign_confidence(final_df, default_type="simulated")
        
        # Output directory save
        output_path = Path("data/processed/rainfall_processed.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        return final_df
