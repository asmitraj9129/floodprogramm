import pandas as pd
import numpy as np

class SourceReliabilityEvaluator:
    """
    Assigns and evaluates source_confidence score [0.0 - 1.0] for multi-source flood risk data.
    Ensures that every record has a transparent confidence score for AI/ML models and Dashboard display.
    """
    
    BASE_CONFIDENCE = {
        "observed": 0.95,
        "historical": 0.98,
        "simulated": 0.85,
        "estimated": 0.82,
        "forecast": 0.78,
        "community_reported": 0.70
    }
    
    @classmethod
    def assign_confidence(cls, df: pd.DataFrame, default_type: str = "simulated") -> pd.DataFrame:
        """
        Ensures data_type and source_confidence columns exist and are accurately calculated.
        """
        df = df.copy()
        
        if "data_type" not in df.columns:
            df["data_type"] = default_type
            
        if "source_confidence" not in df.columns:
            df["source_confidence"] = df["data_type"].map(cls.BASE_CONFIDENCE).fillna(0.80)
            
        # Refine community reported confidence if verification status or report count exists
        if "verification_status" in df.columns:
            mask_verified = df["verification_status"] == "AUTHORITY_VERIFIED"
            mask_confirmed = df["verification_status"] == "COMMUNITY_CONFIRMED"
            mask_unverified = df["verification_status"] == "UNVERIFIED"
            
            df.loc[mask_verified, "source_confidence"] = 0.92
            df.loc[mask_confirmed, "source_confidence"] = 0.80
            df.loc[mask_unverified, "source_confidence"] = 0.60
            
        # Refine forecast confidence based on lead time horizon if present
        if "forecast_horizon_h" in df.columns:
            # Longer forecast horizons have slightly lower confidence
            horizon_penalty = (df["forecast_horizon_h"] - 1).clip(lower=0) * 0.02
            df["source_confidence"] = (df["source_confidence"] - horizon_penalty).clip(lower=0.50, upper=0.98)
            
        df["source_confidence"] = df["source_confidence"].round(3)
        return df

    @classmethod
    def compute_fused_confidence(cls, series_confidences: list) -> float:
        """
        Computes weighted harmonic or root-mean-square mean for combined multi-source observations.
        """
        valid_scores = [c for c in series_confidences if c is not None and not np.isnan(c)]
        if not valid_scores:
            return 0.75
        # Weighted combination favoring lower bounds slightly to be conservative
        return round(float(np.mean(valid_scores) * 0.9 + np.min(valid_scores) * 0.1), 3)
