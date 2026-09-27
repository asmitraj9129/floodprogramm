# BasinPulse AI — Data Dictionary (Member 3)

## Overview
This document defines the schema, units, data types, and confidence weighting for all raw, processed, and fused datasets managed by **Member 3 (Data Engineering & Multi-Source Data Developer)**.

---

## 1. Primary Metadata Tag: `data_type`
Every record across raw, processed, and fused datasets contains a mandatory `data_type` tag to provide full provenance transparency to Member 4 (AI/ML) and Member 1 (Dashboard Visualization):

| `data_type` Value | Description | Default Reliability (`source_confidence`) |
|---|---|---|
| `observed` | Physical gauge/station measurement | 0.90 - 0.95 |
| `estimated` | Spatial interpolation / satellite estimation | 0.80 - 0.85 |
| `forecast` | NWP Numerical model forecast (1h-12h horizon) | 0.75 - 0.80 |
| `historical` | Archived event ground-truth records | 0.95 - 0.98 |
| `simulated` | Software simulation feed (rainfall, stage, soil) | 0.80 - 0.88 |
| `community_reported` | Crowdsourced citizen report | 0.65 - 0.85 |

---

## 2. Raw Datasets Schema (`data/raw/`)

### 2.1 Rainfall Datasets (`rainfall_historical.csv`, `rainfall_simulated.csv`)
- `timestamp`: ISO-8601 UTC string (`YYYY-MM-DDTHH:MM:SSZ`)
- `sub_basin_id`: Sub-basin identifier (`SB-01` to `SB-05`)
- `station_id`: Rain gauge station ID (e.g. `RG-101`)
- `rainfall_mm_1h`: Hourly accumulated rainfall in millimeters
- `intensity_mm_hr`: Instantaneous rainfall rate (mm/hr)
- `data_type`: Metadata provenance (`historical`, `simulated`, `observed`)
- `source_confidence`: Numeric reliability index `[0.0 - 1.0]`

### 2.2 Soil Moisture Dataset (`soil_moisture.csv`)
- `timestamp`: ISO-8601 UTC string
- `sub_basin_id`: Sub-basin identifier (`SB-01` to `SB-05`)
- `soil_moisture_percent`: Volumetric soil moisture saturation percentage `[0.0 - 100.0%]`
- `soil_depth_cm`: Measurement sensor depth (default `10cm`)
- `saturation_status`: Soil state category (`DRY`, `MODERATE`, `SATURATED`, `SUPER_SATURATED`)
- `data_type`: (`observed`, `simulated`)
- `source_confidence`: Numeric reliability index `[0.0 - 1.0]`

### 2.3 River Stage Level Dataset (`water_level.csv`)
- `timestamp`: ISO-8601 UTC string
- `station_id`: Hydrological station ID (e.g. `RIVER-STN-01`)
- `sub_basin_id`: Sub-basin identifier
- `water_level_cm`: Water height above datum in centimeters
- `flow_velocity_m_s`: River surface velocity in meters per second
- `discharge_m3_s`: Estimated stream discharge in cubic meters per second
- `data_type`: (`observed`, `simulated`)
- `source_confidence`: Numeric reliability index `[0.0 - 1.0]`

### 2.4 Weather Forecast Dataset (`rainfall_forecast.csv`)
- `timestamp`: Issued forecast timestamp (ISO-8601 UTC)
- `target_timestamp`: Predicted valid time (ISO-8601 UTC)
- `forecast_horizon_h`: Lead time horizon (1h, 3h, 6h, 12h)
- `sub_basin_id`: Target sub-basin
- `predicted_rainfall_mm`: Predicted hourly rainfall volume
- `probability_heavy_rain`: Probability score `[0.0 - 1.0]`
- `data_type`: `forecast`
- `source_confidence`: Numeric reliability index

### 2.5 Historical Flood Events Dataset (`historical_events.csv`)
- `event_id`: Unique event code (e.g. `HIST-2023-08`)
- `start_time`: Event start timestamp
- `end_time`: Event peak/resolution timestamp
- `sub_basin_id`: Affected sub-basin
- `peak_rainfall_mm_1h`: Peak 1-hour rainfall observed
- `peak_water_level_cm`: Maximum recorded river gauge height
- `flood_severity_class`: Flood severity label (`0`=None, `1`=Minor, `2`=Moderate, `3`=Severe)
- `impacted_villages_count`: Number of flooded habitations
- `data_type`: `historical`
- `source_confidence`: `0.98`

### 2.6 Community Reports Dataset (`sample_reports.json`)
- `report_id`: Report unique ID
- `timestamp`: Submission ISO-8601 timestamp
- `village_id`: Village location identifier
- `sub_basin_id`: Associated sub-basin
- `report_type`: (`water_logging`, `bridge_overtop`, `stream_blockage`, `river_overflow`)
- `severity_level`: User scale `[1 - 5]`
- `latitude`, `longitude`: Spatial coordinates
- `verification_status`: (`UNVERIFIED`, `COMMUNITY_CONFIRMED`, `AUTHORITY_VERIFIED`)
- `data_type`: `community_reported`
- `source_confidence`: Calculated score based on cluster density and verification.

---

## 3. Merged Fused Dataset Schema (`merged_dataset.csv`)
The final consolidated table output by `data_fusion.py` for AI/ML Risk Modeling:

| Column Name | Data Type | Description |
|---|---|---|
| `timestamp` | Datetime (UTC) | Standardized 15-min / 1-hour time index |
| `sub_basin_id` | String | Sub-basin identifier (`SB-01` to `SB-05`) |
| `rainfall_mm_1h` | Float | Accumulated 1-hour rainfall (mm) |
| `rainfall_mm_3h` | Float | Accumulated 3-hour rainfall total (mm) |
| `rainfall_mm_6h` | Float | Accumulated 6-hour rainfall total (mm) |
| `rainfall_intensity_max` | Float | Peak rain intensity in time window |
| `soil_moisture_percent` | Float | Volumetric soil moisture (0-100%) |
| `soil_saturation_index` | Float | Normalized antecedent wetness ratio [0-1] |
| `water_level_cm` | Float | River stage level in cm |
| `water_level_rise_rate_cm_h`| Float | Calculated rate of stage rise (cm/hour) |
| `forecast_rainfall_3h` | Float | NWP forecasted 3-hour rainfall (mm) |
| `forecast_rainfall_6h` | Float | NWP forecasted 6-hour rainfall (mm) |
| `community_severity_score` | Float | Aggregated crowdsourced danger signal index [0-10] |
| `community_report_count` | Integer | Active citizen reports in window |
| `elevation_m` | Float | Sub-basin mean elevation (GIS spatial attribute) |
| `slope_deg` | Float | Sub-basin mean terrain slope (degrees) |
| `distance_to_river_m` | Float | Mean proximity to primary stream channel (meters) |
| `drainage_density_km_km2` | Float | Stream network density ratio |
| `flood_hazard_label` | Integer | Target class for ML training (0=Normal, 1=Watch, 2=Warning, 3=Critical) |
| `data_type` | String | Dominant data source tag (`simulated`, `observed`, etc.) |
| `source_confidence` | Float | Fused multi-source reliability index [0.0 - 1.0] |
