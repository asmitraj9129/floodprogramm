# BasinPulse AI — Member 3 (Data Engineering & Multi-Source Data Fusion)

## 📌 Role & Responsibility
**Role:** Data Engineering & Multi-Source Data Developer  
**Core Responsibility:** *"Data kahan se aayega, clean kaise hoga, and AI ko ek combined dataset kaise milega?"*

Member 3 builds the high-integrity multi-source data pipeline that ingests rainfall, soil moisture, river level, numerical weather forecasts, historical flood events, and crowdsourced community reports. It cleans, handles missing values, aligns timestamps across sub-basins, attaches GIS spatial terrain features, tags provenance (`data_type`), evaluates data reliability (`source_confidence`), and exports a unified fused dataset (`merged_dataset.csv`) for Member 4 (AI/ML) and Member 5 (Backend APIs).

---

## 📁 Repository Structure
```
MEMBER3/
├── data/
│   ├── raw/
│   │   ├── rainfall/
│   │   │   ├── rainfall_historical.csv
│   │   │   └── rainfall_simulated.csv
│   │   ├── soil_moisture/
│   │   │   └── soil_moisture.csv
│   │   ├── river/
│   │   │   └── water_level.csv
│   │   ├── forecast/
│   │   │   └── rainfall_forecast.csv
│   │   ├── historical_flood/
│   │   │   └── historical_events.csv
│   │   └── community_reports/
│   │       └── sample_reports.json
│   │
│   ├── processed/
│   │   ├── rainfall_processed.csv
│   │   ├── soil_processed.csv
│   │   ├── river_processed.csv
│   │   ├── community_signal_processed.csv
│   │   ├── merged_dataset.csv
│   │   └── feature_dictionary.json
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── rainfall_pipeline.py
│   │   ├── soil_pipeline.py
│   │   ├── river_pipeline.py
│   │   ├── historical_pipeline.py
│   │   ├── community_report_pipeline.py
│   │   ├── time_alignment.py
│   │   ├── spatial_alignment.py
│   │   └── data_fusion.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── data_quality.py
│   │   ├── missing_values.py
│   │   ├── outlier_detection.py
│   │   └── source_reliability.py
│   │
│   └── metadata/
│       ├── data_source_registry.json
│       └── data_dictionary.md
│
├── seed_data.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## 🏷️ Mandatory Provenance Tagging: `data_type` & `source_confidence`
Every record generated and processed by Member 3 carries explicit data origin metadata to prevent fake accuracy claims and maintain transparency across the dashboard:

| `data_type` | Description | Default `source_confidence` |
|---|---|---|
| `observed` | Physical gauge or station reading | 0.90 – 0.95 |
| `estimated` | Spatial interpolation / satellite estimation | 0.80 – 0.85 |
| `forecast` | NWP Numerical weather prediction (1h-12h horizon) | 0.75 – 0.80 |
| `historical` | Ground-truth historical flood record | 0.95 – 0.98 |
| `simulated` | Software sensor simulation feed | 0.80 – 0.88 |
| `community_reported` | Crowdsourced citizen report | 0.60 – 0.92 |

---

## ⚙️ Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Data Generation, ETL Fusion & Audit
To generate curated raw data, execute all 5 domain ETL pipelines, merge datasets, and print the quality report:
```bash
python run_pipeline.py
```
Or run `seed_data.py` directly:
```bash
python seed_data.py
```

---

## 🔄 Inter-Member Integration Interfaces

1. **Member 4 (AI/ML & Risk Intelligence)**:
   - Consumes `data/processed/merged_dataset.csv` for training risk models (XGBoost / Random Forest) and running the hybrid rule-based risk engine.
   - Reads `data/processed/feature_dictionary.json` for input feature schemas and target labels (`flood_hazard_label`).

2. **Member 5 (Backend & Simulation Developer)**:
   - Reads `data/raw/` and `data/processed/` to seed the FastAPI database.
   - Uses `data/metadata/data_source_registry.json` for API registry endpoints.

3. **Member 2 (GIS & Terrain Intelligence)**:
   - Provides sub-basin boundary IDs (`SB-01` to `SB-05`), slope angles, elevation, and river proximity metrics consumed by `spatial_alignment.py`.

4. **Member 1 (Authority Dashboard)**:
   - Receives transparent `data_type` and `source_confidence` badges on UI cards (e.g. "Data Freshness", "Confidence Score", "Signal Origin").
