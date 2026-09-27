import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class OutlierDetector:
    """
    Detects sensor noise, physical boundary violations, z-score extreme outliers, and stuck sensors.
    """
    
    # Physical domain bounds
    PHYSICAL_BOUNDS = {
        "rainfall_mm_1h": (0.0, 350.0),        # mm/hour
        "soil_moisture_percent": (0.0, 100.0),  # percentage
        "water_level_cm": (0.0, 3000.0),       # cm stage
        "flow_velocity_m_s": (0.0, 15.0),      # m/s
        "discharge_m3_s": (0.0, 5000.0)        # m3/s
    }

    @classmethod
    def enforce_physical_bounds(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Clips or flags values exceeding absolute hydrometeorological physical limits.
        """
        df = df.copy()
        anomalies_detected = {}
        
        for col, (min_val, max_val) in cls.PHYSICAL_BOUNDS.items():
            if col in df.columns:
                out_of_bounds = ((df[col] < min_val) | (df[col] > max_val)).sum()
                anomalies_detected[col] = int(out_of_bounds)
                df[col] = df[col].clip(lower=min_val, upper=max_val)
                
        return df, anomalies_detected

    @classmethod
    def cap_zscore_outliers(cls, df: pd.DataFrame, cols: List[str], threshold: float = 4.0) -> pd.DataFrame:
        """
        Smooths statistical extreme spikes beyond N standard deviations.
        """
        df = df.copy()
        for col in cols:
            if col in df.columns and len(df) > 10:
                mean = df[col].mean()
                std = df[col].std()
                if std > 1e-6:
                    z_scores = (df[col] - mean).abs() / std
                    mask_spike = z_scores > threshold
                    df.loc[mask_spike, col] = np.sign(df.loc[mask_spike, col] - mean) * threshold * std + mean
        return df

    @classmethod
    def detect_stuck_sensors(cls, df: pd.DataFrame, col: str, window: int = 12, group_col: str = "sub_basin_id") -> pd.DataFrame:
        """
        Flags hardware freeze bugs where sensor outputs constant non-zero value for `window` steps.
        """
        df = df.copy()
        if col in df.columns and group_col in df.columns:
            # Shift check for variance in rolling window
            rolling_std = df.groupby(group_col)[col].transform(lambda s: s.rolling(window=window, min_periods=window).std())
            stuck_mask = (rolling_std == 0.0) & (df[col] > 0)
            # Reduce confidence score for stuck sensor readings
            if "source_confidence" in df.columns:
                df.loc[stuck_mask, "source_confidence"] = (df.loc[stuck_mask, "source_confidence"] * 0.5).round(3)
        return df
