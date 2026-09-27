import pandas as pd
import numpy as np
from typing import Dict, Any

class DataQualityChecker:
    """
    Comprehensive Data Quality & Audit Manager for Member 3.
    """
    
    REQUIRED_FUSED_COLUMNS = [
        "timestamp", "sub_basin_id", "rainfall_mm_1h", "rainfall_mm_3h", 
        "soil_moisture_percent", "water_level_cm", "water_level_rise_rate_cm_h",
        "community_severity_score", "data_type", "source_confidence"
    ]

    @classmethod
    def evaluate_dataset(cls, df: pd.DataFrame, dataset_name: str = "Fused Dataset") -> Dict[str, Any]:
        """
        Executes a rigorous quality audit on any ingested or processed DataFrame.
        """
        total_rows = len(df)
        total_cols = len(df.columns)
        
        missing_per_col = df.isnull().sum().to_dict()
        missing_total = int(df.isnull().sum().sum())
        completeness_pct = round(100.0 * (1.0 - (missing_total / (total_rows * total_cols if total_rows > 0 else 1))), 2)
        
        # Check data_type breakdown
        data_type_counts = df["data_type"].value_counts().to_dict() if "data_type" in df.columns else {}
        
        # Check mean confidence
        mean_confidence = round(float(df["source_confidence"].mean()), 3) if "source_confidence" in df.columns else 0.0
        
        # Column presence check
        missing_required = [col for col in cls.REQUIRED_FUSED_COLUMNS if col not in df.columns]
        
        status = "PASSED" if completeness_pct >= 95.0 and len(missing_required) == 0 else "WARNING"
        if completeness_pct < 80.0:
            status = "FAILED"
            
        report = {
            "dataset_name": dataset_name,
            "status": status,
            "total_records": total_rows,
            "total_columns": total_cols,
            "completeness_percentage": completeness_pct,
            "mean_source_confidence": mean_confidence,
            "data_type_breakdown": data_type_counts,
            "missing_values_per_column": missing_per_col,
            "missing_required_columns": missing_required
        }
        
        return report

    @classmethod
    def print_report_summary(cls, report: Dict[str, Any]):
        """
        Prints formatted quality check summary.
        """
        print(f"\n================ DATA QUALITY REPORT: {report['dataset_name']} ================")
        print(f"Status:                      {report['status']}")
        print(f"Total Records:               {report['total_records']}")
        print(f"Completeness Score:          {report['completeness_percentage']}%")
        print(f"Mean Source Confidence:      {report['mean_source_confidence']}")
        print(f"Data Types Present:          {report['data_type_breakdown']}")
        if report['missing_required_columns']:
            print(f"WARNING Missing Columns:     {report['missing_required_columns']}")
        print("===========================================================================\n")
