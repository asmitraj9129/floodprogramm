import json
import pandas as pd
import numpy as np
from pathlib import Path
from data.validation import SourceReliabilityEvaluator

class CommunityReportPipeline:
    """
    Parses, cleans, filters, and aggregates crowdsourced citizen flood reports.
    Computes sub-basin community hazard signal scores and confidence metrics.
    """
    
    REPORT_SEVERITY_WEIGHTS = {
        "water_logging": 2.0,
        "bridge_overtop": 4.5,
        "stream_blockage": 3.5,
        "river_overflow": 5.0
    }

    def __init__(self, raw_path: str = None):
        self.raw_path = raw_path or "data/raw/community_reports/sample_reports.json"

    def process(self) -> pd.DataFrame:
        if not Path(self.raw_path).exists():
            raise FileNotFoundError(f"Community report file not found at {self.raw_path}")
            
        with open(self.raw_path, 'r', encoding='utf-8') as f:
            raw_reports = json.load(f)
            
        df = pd.DataFrame(raw_reports)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # 1. Assign base report impact weight
        df["report_weight"] = df["report_type"].map(self.REPORT_SEVERITY_WEIGHTS).fillna(2.0)
        df["weighted_severity"] = df["report_weight"] * (df["severity_level"] / 5.0)
        
        # 2. Dynamic Source Reliability Assignment
        df = SourceReliabilityEvaluator.assign_confidence(df, default_type="community_reported")
        
        # 3. Time-window Aggregation (Hourly per sub-basin)
        # Round timestamps to nearest hour
        df["timestamp_hour"] = df["timestamp"].dt.floor("h")
        
        aggregated = df.groupby(["timestamp_hour", "sub_basin_id"]).agg(
            community_report_count=("report_id", "count"),
            community_severity_score=("weighted_severity", lambda x: round(float(np.sum(x)), 2)),
            avg_report_confidence=("source_confidence", "mean")
        ).reset_index()
        
        aggregated.rename(columns={"timestamp_hour": "timestamp"}, inplace=True)
        aggregated["community_severity_score"] = aggregated["community_severity_score"].clip(0.0, 10.0)
        aggregated["data_type"] = "community_reported"
        aggregated["source_confidence"] = aggregated["avg_report_confidence"].round(3)
        aggregated.drop(columns=["avg_report_confidence"], inplace=True)
        
        output_path = Path("data/processed/community_signal_processed.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(output_path, index=False)
        
        return aggregated
