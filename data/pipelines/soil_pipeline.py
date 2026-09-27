import pandas as pd
import numpy as np
from pathlib import Path
from data.validation import MissingValueHandler, OutlierDetector, SourceReliabilityEvaluator

class SoilMoisturePipeline:
    """
    Ingests soil saturation measurements, normalizes soil moisture percentage,
    computes antecedent soil wetness indices, and categorizes saturation risk states.
    """
    
    def __init__(self, raw_path: str = None):
        self.raw_path = raw_path or "data/raw/soil_moisture/soil_moisture.csv"

    def process(self) -> pd.DataFrame:
        if not Path(self.raw_path).exists():
            raise FileNotFoundError(f"Raw soil moisture file not found at {self.raw_path}")
            
        df = pd.read_csv(self.raw_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values(by=["sub_basin_id", "timestamp"]).reset_index(drop=True)
        
        # 1. Physical limits and outlier handling
        df, _ = OutlierDetector.enforce_physical_bounds(df)
        df = OutlierDetector.detect_stuck_sensors(df, col="soil_moisture_percent", window=12)
        
        # 2. Impute continuous values
        df = MissingValueHandler.impute_continuous_signal(df, cols=["soil_moisture_percent"], group_col="sub_basin_id")
        
        # 3. Feature Engineering: Saturation state & Antecedent Soil Wetness Index (0.0 to 1.0)
        df["soil_saturation_index"] = (df["soil_moisture_percent"] / 100.0).clip(0.0, 1.0).round(3)
        
        conditions = [
            (df["soil_moisture_percent"] < 40.0),
            (df["soil_moisture_percent"] >= 40.0) & (df["soil_moisture_percent"] < 70.0),
            (df["soil_moisture_percent"] >= 70.0) & (df["soil_moisture_percent"] < 88.0),
            (df["soil_moisture_percent"] >= 88.0)
        ]
        choices = ["DRY", "MODERATE", "SATURATED", "SUPER_SATURATED"]
        df["saturation_status"] = np.select(conditions, choices, default="MODERATE")
        
        # 4. Source Reliability Tagging
        df = SourceReliabilityEvaluator.assign_confidence(df, default_type="observed")
        
        output_path = Path("data/processed/soil_processed.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return df
