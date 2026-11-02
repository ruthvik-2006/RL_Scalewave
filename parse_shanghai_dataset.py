import os
import glob
import json
import numpy as np
import pandas as pd

def process_shanghai_excel_files(dataset_dir=".", output_json="shanghai_workload_trace.json"):
    """
    Parses Shanghai Telecom Excel files (.xlsx) directly matching the columns:
    [month, date, start time, end time, latitude, longitude, user id]
    """
    # Look for .xlsx files in the specified folder
    excel_files = glob.glob(os.path.join(dataset_dir, "*.xlsx"))
    
    if not excel_files:
        print(f"[!] No .xlsx files found in '{dataset_dir}'. Generating fallback workload trace...")
        np.random.seed(42)
        base_arrivals = np.random.exponential(scale=0.35, size=3600).tolist()
        trace_data = {
            "base_station_id": 4,
            "total_requests": len(base_arrivals),
            "inter_arrival_seconds": base_arrivals
        }
        with open(output_json, "w") as f:
            json.dump(trace_data, f, indent=2)
        print(f"[✓] Trace file saved to '{output_json}'")
        return

    # Select the first Excel file (e.g., data_6.16.15.xlsx)
    target_file = excel_files[0]
    print(f"[+] Processing real Shanghai Telecom Excel file: '{target_file}'...")

    # Load Excel sheet using pandas and openpyxl
    df = pd.read_excel(target_file, engine='openpyxl')
    
    # Clean up column names (strip whitespace and convert to lowercase)
    df.columns = [str(col).strip().lower() for col in df.columns]
    print(f"[+] Identified columns: {list(df.columns)}")

    # Clean rows missing location or time values
    df = df.dropna(subset=['latitude', 'longitude', 'start time'])

    # Convert 'start time' column to datetime objects
    df['start_time_dt'] = pd.to_datetime(df['start time'], errors='coerce')
    df = df.dropna(subset=['start_time_dt'])

    # Group coordinates (latitude, longitude) into unique Base Station IDs
    df['bs_id'] = df.groupby(['latitude', 'longitude']).ngroup()

    # Find the Base Station with highest traffic volume (to capture peak burstiness)
    top_bs_id = df['bs_id'].value_counts().index[0]
    print(f"[+] Selected Base Station ID {top_bs_id} with maximum request traffic.")

    # Filter records for selected base station and sort by timestamp
    bs_df = df[df['bs_id'] == top_bs_id].sort_values(by='start_time_dt')

    # Compute inter-arrival time deltas in seconds
    bs_df['inter_arrival_sec'] = bs_df['start_time_dt'].diff().dt.total_seconds()

    # Filter valid positive intervals and clip minimum time delta to 0.01s
    inter_arrivals = bs_df['inter_arrival_sec'].dropna().clip(lower=0.01).tolist()

    # Take an hour-long trace window (~3,600 requests)
    sample_trace = inter_arrivals[:3600]

    trace_data = {
        "base_station_id": int(top_bs_id),
        "total_requests": len(sample_trace),
        "inter_arrival_seconds": sample_trace
    }

    with open(output_json, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"[✓] Successfully parsed Excel dataset! Saved {len(sample_trace)} request trace steps to '{output_json}'")

if __name__ == "__main__":
    process_shanghai_excel_files()