import pandas as pd
import numpy as np
from pathlib import Path
from data.validation import MissingValueHandler, OutlierDetector, SourceReliabilityEvaluator

class RiverHydrologyPipeline:
    """
    Ingests river stage height (cm), discharge rate, computes rate-of-rise (cm/hr),
    identifies rapid stage surge anomalies, and tags stage alert thresholds.
    """
    
    WARNING_STAGE_CM = 150.0
    DANGER_STAGE_CM = 220.0
    OVERTOP_STAGE_CM = 280.0

    def __init__(self, raw_path: str = None):
        self.raw_path = raw_path or "data/raw/river/water_level.csv"

    def process(self) -> pd.DataFrame:
        if not Path(self.raw_path).exists():
            raise FileNotFoundError(f"Raw river stage file not found at {self.raw_path}")
            
        df = pd.read_csv(self.raw_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values(by=["sub_basin_id", "timestamp"]).reset_index(drop=True)
        
        # 1. Clean outliers and check bounds
        df, _ = OutlierDetector.enforce_physical_bounds(df)
        df = MissingValueHandler.impute_continuous_signal(
            df, cols=["water_level_cm", "flow_velocity_m_s", "discharge_m3_s"], group_col="sub_basin_id"
        )
        
        # 2. Derivative Feature Engineering: Rate of rise (cm/hour)
        processed_records = []
        for sub_id, group in df.groupby("sub_basin_id"):
            group = group.copy().sort_values("timestamp")
            
            # Difference in water level over time step
            group["water_level_rise_rate_cm_h"] = group["water_level_cm"].diff().fillna(0.0).round(2)
            # Acceleration of water rise
            group["water_level_acceleration"] = group["water_level_rise_rate_cm_h"].diff().fillna(0.0).round(2)
            
            processed_records.append(group)
            
        final_df = pd.concat(processed_records, ignore_index=True)
        
        # 3. Categorical Threshold Hazard Flags
        conditions = [
            (final_df["water_level_cm"] < self.WARNING_STAGE_CM),
            (final_df["water_level_cm"] >= self.WARNING_STAGE_CM) & (final_df["water_level_cm"] < self.DANGER_STAGE_CM),
            (final_df["water_level_cm"] >= self.DANGER_STAGE_CM) & (final_df["water_level_cm"] < self.OVERTOP_STAGE_CM),
            (final_df["water_level_cm"] >= self.OVERTOP_STAGE_CM)
        ]
        choices = ["NORMAL", "WATCH", "WARNING", "CRITICAL_OVERTOP"]
        final_df["river_stage_status"] = np.select(conditions, choices, default="NORMAL")
        
        # 4. Reliability Tagging
        final_df = SourceReliabilityEvaluator.assign_confidence(final_df, default_type="observed")
        
        output_path = Path("data/processed/river_processed.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        return final_df
