import pandas as pd
import numpy as np
from typing import List, Dict

class MissingValueHandler:
    """
    Handles missing value identification, imputation strategies, and quality reporting for hydrological data.
    """
    
    @staticmethod
    def inspect_missing(df: pd.DataFrame) -> Dict[str, float]:
        """
        Returns a dictionary of missing percentage per column.
        """
        missing_count = df.isnull().sum()
        total_rows = len(df)
        if total_rows == 0:
            return {}
        return {col: round((count / total_rows) * 100, 2) for col, count in missing_count.items()}

    @staticmethod
    def impute_rainfall(df: pd.DataFrame, col: str = "rainfall_mm_1h", group_col: str = "sub_basin_id") -> pd.DataFrame:
        """
        Imputes missing rainfall values using zero-fill or time-based forward fill.
        Rainfall is discrete; NaN during sensor outages is safely filled with 0.0 or group mean.
        """
        df = df.copy()
        if col in df.columns:
            # First try forward fill per group for brief gaps, then zero fill
            df[col] = df.groupby(group_col)[col].ffill(limit=2).fillna(0.0)
            df[col] = df[col].clip(lower=0.0)
        return df

    @staticmethod
    def impute_continuous_signal(df: pd.DataFrame, cols: List[str], group_col: str = "sub_basin_id") -> pd.DataFrame:
        """
        Imputes continuous hydrological variables (soil moisture, river stage, flow velocity)
        using linear time interpolation per sub-basin, followed by bfill/ffill.
        """
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = df.groupby(group_col)[col].transform(lambda g: g.interpolate(method='linear', limit=4))
                df[col] = df.groupby(group_col)[col].ffill().bfill()
        return df
