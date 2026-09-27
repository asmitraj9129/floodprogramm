#!/usr/bin/env python3
"""
BasinPulse AI - Member 3 Runner CLI Script
Executes data generation, ETL pipelines, multi-source data fusion, quality audits, and seed exports.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add current directory to python path
sys.path.insert(0, str(Path(__file__).parent))

from seed_data import generate_raw_sample_data
from data.pipelines.data_fusion import DataFusionEngine
from data.validation.data_quality import DataQualityChecker

def main():
    parser = argparse.ArgumentParser(description="BasinPulse AI - Member 3 Data Engineering & Multi-Source Fusion CLI")
    parser.add_argument("--generate-raw", action="store_true", help="Generate fresh curated sample raw datasets in data/raw/")
    parser.add_argument("--run-fusion", action="store_true", help="Execute ETL pipelines and generate data/processed/merged_dataset.csv")
    parser.add_argument("--audit", action="store_true", help="Run comprehensive data quality audit")
    
    args = parser.parse_args()
    
    # If no flags passed, run full workflow
    if not (args.generate_raw or args.run_fusion or args.audit):
        args.generate_raw = True
        args.run_fusion = True
        args.audit = True
        
    if args.generate_raw:
        generate_raw_sample_data()
        
    if args.run_fusion:
        print("\n========================================================")
        print("          BASINPULSE AI - MEMBER 3 DATA FUSION          ")
        print("========================================================\n")
        engine = DataFusionEngine()
        df = engine.run_full_fusion()
        
        print("\n[SUCCESS] Processed datasets written to data/processed/:")
        print("  - rainfall_processed.csv")
        print("  - soil_processed.csv")
        print("  - river_processed.csv")
        print("  - community_signal_processed.csv")
        print("  - merged_dataset.csv")
        print("  - feature_dictionary.json")

if __name__ == "__main__":
    main()
