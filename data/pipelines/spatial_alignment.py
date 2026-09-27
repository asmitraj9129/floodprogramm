import pandas as pd
import numpy as np

class SpatialAlignmentEngine:
    """
    Enriches sub-basin time series with static spatial attributes derived from GIS (Member 2).
    Maps station points & catchment polygons to unified sub-basin feature vectors.
    """
    
    # Sub-basin Spatial Lookup Table (matches Member 2 GeoJSON boundaries)
    SUB_BASIN_SPATIAL_FEATURES = {
        "SB-01": {"elevation_m": 480.0, "slope_deg": 18.5, "distance_to_river_m": 120.0, "drainage_density_km_km2": 2.85},
        "SB-02": {"elevation_m": 320.0, "slope_deg": 12.0, "distance_to_river_m": 85.0,  "drainage_density_km_km2": 2.40},
        "SB-03": {"elevation_m": 180.0, "slope_deg": 4.5,  "distance_to_river_m": 30.0,  "drainage_density_km_km2": 3.60},
        "SB-04": {"elevation_m": 95.0,  "slope_deg": 2.1,  "distance_to_river_m": 15.0,  "drainage_density_km_km2": 3.95},
        "SB-05": {"elevation_m": 390.0, "slope_deg": 14.8, "distance_to_river_m": 95.0,  "drainage_density_km_km2": 2.20}
    }

    @classmethod
    def attach_gis_features(cls, df: pd.DataFrame, sub_basin_col: str = "sub_basin_id") -> pd.DataFrame:
        """
        Appends spatial terrain intelligence columns to the dataset.
        """
        df = df.copy()
        if sub_basin_col not in df.columns:
            df[sub_basin_col] = "SB-03"
            
        spatial_df = pd.DataFrame.from_dict(cls.SUB_BASIN_SPATIAL_FEATURES, orient="index").reset_index()
        spatial_df.rename(columns={"index": sub_basin_col}, inplace=True)
        
        merged_df = pd.merge(df, spatial_df, on=sub_basin_col, how="left")
        return merged_df
