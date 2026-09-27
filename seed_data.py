import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Fix random seed for reproducible hackathon demo data
np.random.seed(42)
random.seed(42)

def generate_raw_sample_data():
    """
    Generates realistic curated raw sample/simulated datasets for BasinPulse AI (Member 3).
    Covers a 72-hour timeline with a severe flash flood storm event occurring between hours 36 and 54.
    """
    print(">>> Generating curated raw multi-source datasets in data/raw/...")
    
    base_time = datetime(2026, 9, 24, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(72)]
    sub_basins = ["SB-01", "SB-02", "SB-03", "SB-04", "SB-05"]
    
    # 1. Generate Historical Rain & Simulated Real-time Rain
    rain_hist_rows = []
    rain_sim_rows = []
    
    for sb in sub_basins:
        # Multiplier per sub-basin to represent spatial rain gradient
        sb_factor = {"SB-01": 1.3, "SB-02": 1.1, "SB-03": 1.0, "SB-04": 0.8, "SB-05": 1.2}[sb]
        
        for idx, t in enumerate(timestamps):
            t_str = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Normal base rain: 0-5mm
            base_rain = np.random.uniform(0.0, 3.0)
            
            # Flash flood storm surge event during hours 36 to 52
            if 36 <= idx <= 52:
                storm_intensity = np.sin((idx - 36) / 16.0 * np.pi) * 75.0 * sb_factor
                rain_val = round(base_rain + storm_intensity + np.random.uniform(0, 10), 2)
            else:
                rain_val = round(base_rain, 2)
                
            # Historical dataset (first 48 hours)
            if idx < 48:
                rain_hist_rows.append({
                    "timestamp": t_str,
                    "sub_basin_id": sb,
                    "station_id": f"RG-{sb[-2:]}-01",
                    "rainfall_mm_1h": rain_val,
                    "intensity_mm_hr": round(rain_val * np.random.uniform(0.9, 1.2), 2),
                    "data_type": "historical",
                    "source_confidence": 0.95
                })
                
            # Simulated real-time dataset (hours 24 to 72)
            if idx >= 24:
                rain_sim_rows.append({
                    "timestamp": t_str,
                    "sub_basin_id": sb,
                    "station_id": f"SIM-RG-{sb[-2:]}",
                    "rainfall_mm_1h": rain_val,
                    "intensity_mm_hr": round(rain_val * 1.1, 2),
                    "data_type": "simulated",
                    "source_confidence": 0.85
                })

    Path("data/raw/rainfall").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rain_hist_rows).to_csv("data/raw/rainfall/rainfall_historical.csv", index=False)
    pd.DataFrame(rain_sim_rows).to_csv("data/raw/rainfall/rainfall_simulated.csv", index=False)
    print(" -> Rainfall historical and simulated CSVs created.")

    # 2. Generate Soil Moisture Dataset
    soil_rows = []
    for sb in sub_basins:
        current_sm = 35.0 # initial dry soil %
        for idx, t in enumerate(timestamps):
            t_str = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Rain input boost
            if 36 <= idx <= 52:
                current_sm = min(98.0, current_sm + np.random.uniform(3.5, 6.0))
            else:
                current_sm = max(25.0, current_sm - np.random.uniform(0.2, 0.5))
                
            sm_val = round(current_sm + np.random.uniform(-1.0, 1.0), 1)
            sm_val = float(np.clip(sm_val, 0.0, 100.0))
            
            status = "DRY" if sm_val < 40 else ("MODERATE" if sm_val < 70 else ("SATURATED" if sm_val < 88 else "SUPER_SATURATED"))
            
            soil_rows.append({
                "timestamp": t_str,
                "sub_basin_id": sb,
                "soil_moisture_percent": sm_val,
                "soil_depth_cm": 10,
                "saturation_status": status,
                "data_type": "observed",
                "source_confidence": 0.90
            })

    Path("data/raw/soil_moisture").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(soil_rows).to_csv("data/raw/soil_moisture/soil_moisture.csv", index=False)
    print(" -> Soil moisture CSV created.")

    # 3. Generate River Hydrology Water Level Dataset
    river_rows = []
    for sb in sub_basins:
        base_stage = 45.0 # cm
        for idx, t in enumerate(timestamps):
            t_str = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Stage rise lagged after rain surge (hours 38 to 58)
            if 38 <= idx <= 58:
                surge = np.sin((idx - 38) / 20.0 * np.pi) * 230.0
                water_stage = base_stage + surge + np.random.uniform(0, 8)
            else:
                water_stage = base_stage + np.random.uniform(-2, 3)
                
            stage_val = round(max(10.0, water_stage), 1)
            velocity = round(0.5 + (stage_val / 250.0) * 3.5, 2)
            discharge = round(15.0 + (stage_val / 250.0) * 850.0, 1)
            
            river_rows.append({
                "timestamp": t_str,
                "station_id": f"RIVER-STN-{sb[-2:]}",
                "sub_basin_id": sb,
                "water_level_cm": stage_val,
                "flow_velocity_m_s": velocity,
                "discharge_m3_s": discharge,
                "data_type": "observed",
                "source_confidence": 0.92
            })

    Path("data/raw/river").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(river_rows).to_csv("data/raw/river/water_level.csv", index=False)
    print(" -> River water level CSV created.")

    # 4. Generate Numerical Weather Forecast Dataset
    forecast_rows = []
    for sb in sub_basins:
        for idx in range(30, 48): # issued leading up to and during storm
            t_issued = timestamps[idx]
            for horizon in [1, 3, 6, 12]:
                t_target = t_issued + timedelta(hours=horizon)
                target_idx = idx + horizon
                
                if target_idx < len(timestamps) and 36 <= target_idx <= 52:
                    pred_rain = round(np.random.uniform(25.0, 65.0), 2)
                    prob_heavy = round(np.random.uniform(0.75, 0.95), 2)
                else:
                    pred_rain = round(np.random.uniform(0.0, 5.0), 2)
                    prob_heavy = round(np.random.uniform(0.05, 0.20), 2)
                    
                forecast_rows.append({
                    "timestamp": t_issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "target_timestamp": t_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "forecast_horizon_h": horizon,
                    "sub_basin_id": sb,
                    "predicted_rainfall_mm": pred_rain,
                    "probability_heavy_rain": prob_heavy,
                    "data_type": "forecast",
                    "source_confidence": round(0.80 - (horizon - 1) * 0.02, 2)
                })

    Path("data/raw/forecast").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forecast_rows).to_csv("data/raw/forecast/rainfall_forecast.csv", index=False)
    print(" -> Rainfall forecast CSV created.")

    # 5. Generate Historical Flood Events Dataset
    hist_events = [
        {
            "event_id": "HIST-2023-08",
            "start_time": "2023-08-14T04:00:00Z",
            "end_time": "2023-08-15T18:00:00Z",
            "sub_basin_id": "SB-03",
            "peak_rainfall_mm_1h": 72.5,
            "peak_water_level_cm": 265.0,
            "flood_severity_class": 3,
            "impacted_villages_count": 5,
            "data_type": "historical",
            "source_confidence": 0.98
        },
        {
            "event_id": "HIST-2024-07",
            "start_time": "2024-07-21T10:00:00Z",
            "end_time": "2024-07-22T08:00:00Z",
            "sub_basin_id": "SB-01",
            "peak_rainfall_mm_1h": 54.0,
            "peak_water_level_cm": 195.0,
            "flood_severity_class": 2,
            "impacted_villages_count": 2,
            "data_type": "historical",
            "source_confidence": 0.98
        }
    ]
    Path("data/raw/historical_flood").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist_events).to_csv("data/raw/historical_flood/historical_events.csv", index=False)
    print(" -> Historical flood events CSV created.")

    # 6. Generate Crowdsourced Community Reports JSON
    community_reports = []
    villages = [
        {"id": "V-01", "name": "Rampur Ghat", "sub": "SB-03", "lat": 30.1245, "lon": 78.1452},
        {"id": "V-02", "name": "Devnagar", "sub": "SB-03", "lat": 30.1310, "lon": 78.1520},
        {"id": "V-03", "name": "Highland Ridge", "sub": "SB-01", "lat": 30.1890, "lon": 78.0980},
        {"id": "V-04", "name": "Bridge Colony B17", "sub": "SB-03", "lat": 30.1190, "lon": 78.1380}
    ]
    
    report_types = ["water_logging", "bridge_overtop", "stream_blockage", "river_overflow"]
    verifications = ["AUTHORITY_VERIFIED", "COMMUNITY_CONFIRMED", "UNVERIFIED"]
    
    rep_count = 1
    for idx in range(38, 54): # Peak storm window
        t_str = timestamps[idx].strftime("%Y-%m-%dT%H:%M:%SZ")
        for v in villages:
            if random.random() > 0.4:
                rtype = random.choice(report_types)
                sev = random.randint(3, 5)
                v_stat = random.choice(verifications)
                
                community_reports.append({
                    "report_id": f"REP-{rep_count:04d}",
                    "timestamp": t_str,
                    "village_id": v["id"],
                    "village_name": v["name"],
                    "sub_basin_id": v["sub"],
                    "report_type": rtype,
                    "severity_level": sev,
                    "latitude": v["lat"],
                    "longitude": v["lon"],
                    "verification_status": v_stat,
                    "data_type": "community_reported",
                    "source_confidence": 0.92 if v_stat == "AUTHORITY_VERIFIED" else (0.80 if v_stat == "COMMUNITY_CONFIRMED" else 0.60)
                })
                rep_count += 1
                
    Path("data/raw/community_reports").mkdir(parents=True, exist_ok=True)
    with open("data/raw/community_reports/sample_reports.json", "w", encoding="utf-8") as f:
        json.dump(community_reports, f, indent=2)
    print(" -> Community reports JSON created.")
    
    print(">>> All raw sample datasets successfully generated!\n")

if __name__ == "__main__":
    generate_raw_sample_data()
    
    # Import and run DataFusionEngine
    from data.pipelines.data_fusion import DataFusionEngine
    engine = DataFusionEngine()
    engine.run_full_fusion()
