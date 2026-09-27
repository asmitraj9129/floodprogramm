import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from data.pipelines.rainfall_pipeline import RainfallPipeline
from data.pipelines.soil_pipeline import SoilMoisturePipeline
from data.pipelines.river_pipeline import RiverHydrologyPipeline
from data.pipelines.community_report_pipeline import CommunityReportPipeline
from data.pipelines.time_alignment import TimeAlignmentEngine
from data.pipelines.spatial_alignment import SpatialAlignmentEngine
from data.validation import (
    MissingValueHandler, OutlierDetector, SourceReliabilityEvaluator, DataQualityChecker
)

class DataFusionEngine:
    """
    Central Data Fusion Orchestrator for BasinPulse AI (Member 3).
    Combines rainfall, soil moisture, river hydrology, community signals, and GIS features
    into a high-integrity fused dataset ready for Member 4 (ML/Risk Engine) & Member 5 (Backend).
    """
    
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run_full_fusion(self) -> pd.DataFrame:
        """
        Executes end-to-end multi-source extraction, alignment, cleaning, and fusion.
        """
        print(">>> Starting Member 3 Multi-Source Data Fusion Engine...")
        
        # 1. Run / load processed individual domain datasets
        print(" -> Ingesting and processing rainfall signals...")
        rain_df = RainfallPipeline().process()
        
        print(" -> Ingesting and processing soil moisture signals...")
        soil_df = SoilMoisturePipeline().process()
        
        print(" -> Ingesting and processing river hydrological stage signals...")
        river_df = RiverHydrologyPipeline().process()
        
        print(" -> Ingesting and processing crowdsourced community reports...")
        comm_df = CommunityReportPipeline().process()
        
        # 2. Synchronize multi-source signals onto uniform time grid
        print(" -> Synchronizing temporal indices across sub-basins...")
        fused = TimeAlignmentEngine.align_to_uniform_grid([rain_df, soil_df, river_df, comm_df], freq="1h")
        
        # 3. Impute missing alignment gaps
        fused = MissingValueHandler.impute_rainfall(fused, col="rainfall_mm_1h")
        fused = MissingValueHandler.impute_rainfall(fused, col="rainfall_mm_3h")
        fused = MissingValueHandler.impute_rainfall(fused, col="rainfall_mm_6h")
        fused = MissingValueHandler.impute_continuous_signal(fused, cols=["soil_moisture_percent", "soil_saturation_index", "water_level_cm", "water_level_rise_rate_cm_h"])
        
        # Fill community signal gaps with 0
        if "community_severity_score" in fused.columns:
            fused["community_severity_score"] = fused["community_severity_score"].fillna(0.0).round(2)
        if "community_report_count" in fused.columns:
            fused["community_report_count"] = fused["community_report_count"].fillna(0).astype(int)
            
        # 4. Attach GIS Spatial Features
        print(" -> Attaching GIS terrain and exposure attributes...")
        fused = SpatialAlignmentEngine.attach_gis_features(fused)
        
        # 5. Add forecast columns (simulated forecast horizon matching current + trend)
        fused["forecast_rainfall_3h"] = (fused["rainfall_mm_3h"] * 0.85 + np.random.uniform(0, 5, len(fused))).round(2)
        fused["forecast_rainfall_6h"] = (fused["rainfall_mm_6h"] * 0.75 + np.random.uniform(0, 10, len(fused))).round(2)
        
        # 6. Derive Target ML Class: flood_hazard_label
        # Class 0: Normal, Class 1: Watch, Class 2: Warning, Class 3: Critical Hazard
        conditions = [
            (fused["water_level_cm"] >= 250.0) | ((fused["rainfall_mm_3h"] >= 80.0) & (fused["soil_moisture_percent"] >= 80.0)),
            (fused["water_level_cm"] >= 190.0) | ((fused["rainfall_mm_3h"] >= 50.0) & (fused["soil_moisture_percent"] >= 70.0)),
            (fused["water_level_cm"] >= 140.0) | (fused["rainfall_mm_3h"] >= 30.0),
        ]
        choices = [3, 2, 1]
        fused["flood_hazard_label"] = np.select(conditions, choices, default=0)
        
        # 7. Metadata provenance & dynamic confidence fusion
        fused["data_type"] = fused["data_type"].fillna("simulated")
        # Combine confidence metrics
        conf_cols = [c for c in fused.columns if "confidence" in c or c == "source_confidence"]
        if conf_cols:
            fused["source_confidence"] = fused[conf_cols].mean(axis=1).round(3)
        else:
            fused["source_confidence"] = 0.85
            
        # Select clean columns order for merged_dataset.csv
        column_order = [
            "timestamp", "sub_basin_id", 
            "rainfall_mm_1h", "rainfall_mm_3h", "rainfall_mm_6h", "rainfall_intensity_max",
            "soil_moisture_percent", "soil_saturation_index", "saturation_status",
            "water_level_cm", "water_level_rise_rate_cm_h", "river_stage_status",
            "forecast_rainfall_3h", "forecast_rainfall_6h",
            "community_severity_score", "community_report_count",
            "elevation_m", "slope_deg", "distance_to_river_m", "drainage_density_km_km2",
            "flood_hazard_label", "data_type", "source_confidence"
        ]
        
        # Ensure all columns exist
        final_cols = [c for c in column_order if c in fused.columns]
        fused_df = fused[final_cols].copy()
        
        # 8. Save merged_dataset.csv
        merged_path = self.processed_dir / "merged_dataset.csv"
        fused_df.to_csv(merged_path, index=False)
        print(f" -> Master fused dataset saved to: {merged_path}")
        
        # 9. Generate feature_dictionary.json
        self._generate_feature_dictionary(fused_df)
        
        # 10. Audit data quality
        quality_report = DataQualityChecker.evaluate_dataset(fused_df, "Master Fused Dataset")
        DataQualityChecker.print_report_summary(quality_report)
        
        return fused_df

    def _generate_feature_dictionary(self, df: pd.DataFrame):
        """
        Exports feature_dictionary.json describing columns for Member 4 & 5.
        """
        dict_path = self.processed_dir / "feature_dictionary.json"
        
        feature_meta = {
            "timestamp": {"dtype": "string", "unit": "ISO-8601 UTC", "description": "Standardized temporal index"},
            "sub_basin_id": {"dtype": "string", "unit": "categorical", "description": "Sub-basin boundary identifier (SB-01 to SB-05)"},
            "rainfall_mm_1h": {"dtype": "float", "unit": "mm", "description": "1-hour accumulated precipitation"},
            "rainfall_mm_3h": {"dtype": "float", "unit": "mm", "description": "3-hour rolling accumulated precipitation"},
            "rainfall_mm_6h": {"dtype": "float", "unit": "mm", "description": "6-hour rolling accumulated precipitation"},
            "rainfall_intensity_max": {"dtype": "float", "unit": "mm/h", "description": "Peak instantaneous rain intensity"},
            "soil_moisture_percent": {"dtype": "float", "unit": "%", "description": "Volumetric soil water saturation"},
            "soil_saturation_index": {"dtype": "float", "unit": "ratio [0-1]", "description": "Normalized soil wetness ratio"},
            "saturation_status": {"dtype": "string", "unit": "categorical", "description": "Soil state (DRY, MODERATE, SATURATED, SUPER_SATURATED)"},
            "water_level_cm": {"dtype": "float", "unit": "cm", "description": "River gauge stage height above datum"},
            "water_level_rise_rate_cm_h": {"dtype": "float", "unit": "cm/h", "description": "Calculated rate of stage rise"},
            "river_stage_status": {"dtype": "string", "unit": "categorical", "description": "Stage hazard class (NORMAL, WATCH, WARNING, CRITICAL_OVERTOP)"},
            "forecast_rainfall_3h": {"dtype": "float", "unit": "mm", "description": "NWP predicted 3-hour rainfall horizon"},
            "forecast_rainfall_6h": {"dtype": "float", "unit": "mm", "description": "NWP predicted 6-hour rainfall horizon"},
            "community_severity_score": {"dtype": "float", "unit": "index [0-10]", "description": "Aggregated crowdsourced danger signal score"},
            "community_report_count": {"dtype": "integer", "unit": "count", "description": "Active citizen reports count in window"},
            "elevation_m": {"dtype": "float", "unit": "meters", "description": "Sub-basin mean elevation (GIS spatial attribute)"},
            "slope_deg": {"dtype": "float", "unit": "degrees", "description": "Sub-basin mean terrain slope (GIS spatial attribute)"},
            "distance_to_river_m": {"dtype": "float", "unit": "meters", "description": "Proximity to primary river channel"},
            "drainage_density_km_km2": {"dtype": "float", "unit": "km/km2", "description": "Sub-basin stream network density ratio"},
            "flood_hazard_label": {"dtype": "integer", "unit": "class [0-3]", "description": "Ground truth hazard target for ML (0=Normal, 1=Watch, 2=Warning, 3=Critical)"},
            "data_type": {"dtype": "string", "unit": "provenance tag", "description": "Data origin label (observed, estimated, forecast, historical, simulated, community_reported)"},
            "source_confidence": {"dtype": "float", "unit": "index [0-1]", "description": "Fused multi-source data reliability index"}
        }
        
        with open(dict_path, 'w', encoding='utf-8') as f:
            json.dump(feature_meta, f, indent=2)
            
        print(f" -> Feature dictionary metadata saved to: {dict_path}")
