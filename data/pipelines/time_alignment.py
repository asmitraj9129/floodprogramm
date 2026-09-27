import pandas as pd
import numpy as np
from typing import List, Dict

class TimeAlignmentEngine:
    """
    Synchronizes multi-frequency time series (rainfall, river level, soil moisture, forecast)
    onto a unified temporal grid per sub-basin without data leakage.
    """
    
    @staticmethod
    def align_to_uniform_grid(
        dfs: List[pd.DataFrame], 
        freq: str = "1h", 
        sub_basin_col: str = "sub_basin_id", 
        time_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Merges list of sub-basin DataFrames on a standardized uniform time index grid.
        """
        if not dfs:
            raise ValueError("No DataFrames provided for time alignment.")
            
        # Standardize timestamps
        aligned_dfs = []
        for df in dfs:
            d = df.copy()
            d[time_col] = pd.to_datetime(d[time_col], utc=True)
            aligned_dfs.append(d)
            
        # Find global min and max timestamps
        min_time = min(d[time_col].min() for d in aligned_dfs)
        max_time = max(d[time_col].max() for d in aligned_dfs)
        sub_basins = sorted(list(set().union(*[d[sub_basin_col].unique() for d in aligned_dfs if sub_basin_col in d.columns])))
        
        # Create full grid
        time_range = pd.date_range(start=min_time, end=max_time, freq=freq, tz="UTC")
        grid_tuples = [(t, sb) for sb in sub_basins for t in time_range]
        master_grid = pd.DataFrame(grid_tuples, columns=[time_col, sub_basin_col])
        
        # Sequentially merge each dataset onto master grid
        fused_df = master_grid.copy()
        for d in aligned_dfs:
            # Drop duplicate columns before merge to avoid _x, _y collision except merge keys
            cols_to_use = [c for c in d.columns if c not in fused_df.columns or c in [time_col, sub_basin_col]]
            fused_df = pd.merge(fused_df, d[cols_to_use], on=[time_col, sub_basin_col], how="left")
            
        # Sort chronologically
        fused_df = fused_df.sort_values(by=[sub_basin_col, time_col]).reset_index(drop=True)
        return fused_df
