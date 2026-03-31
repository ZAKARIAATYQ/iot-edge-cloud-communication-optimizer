"""
multi_real_utils.py
-------------------
Data loading, temporal alignment, and validation utilities for the
real multi-sensor simulation (5 Moroccan cities).

Each CSV in 5_SENSOR/ has columns: valid_time, t2m, latitude, longitude.
Temperature is in Kelvin and gets converted to Celsius.

This module does NOT modify any existing files or modules.
"""

import os
import sys
import numpy as np
import pandas as pd

# Sensor filenames and their human-readable city names (column order)
SENSOR_FILES = [
    ("AZILALSENSOR.csv",          "Azilal"),
    ("BENIMELLALSENSOR.csv",      "Beni Mellal"),
    ("FQUIHBENSALAHSENSOR.csv",   "Fquih Ben Salah"),
    ("KHENIFRASENSOR.csv",        "Khenifra"),
    ("MARAKKECHSENSOR.csv",       "Marrakech"),
]

# Default path to the 5_SENSOR directory (relative to project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SENSOR_DIR = os.path.join(_PROJECT_ROOT, "5_SENSOR")


def load_real_multi_sensor_data(sensor_dir=None):
    """
    Load, align, and validate the 5 real city sensor datasets.

    Steps:
        1. Read each CSV and extract (valid_time, t2m).
        2. Convert t2m from Kelvin to Celsius.
        3. Inner-merge on valid_time to guarantee identical timestamps.
        4. Sort chronologically.
        5. Drop any rows with NaN after merge.
        6. Validate: no NaNs, shape == (time_steps, 5).

    Returns:
        data_matrix (np.ndarray): shape (time_steps, 5), each column = one city.
        city_names (list[str]): ordered city names matching columns.
        timestamps (pd.DatetimeIndex): aligned timestamps.
    """
    if sensor_dir is None:
        sensor_dir = DEFAULT_SENSOR_DIR

    city_names = []
    merged_df = None

    for filename, city_name in SENSOR_FILES:
        filepath = os.path.join(sensor_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Sensor file not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=["valid_time"])
        # Keep only time and temperature
        df = df[["valid_time", "t2m"]].copy()
        # Convert Kelvin -> Celsius
        if df["t2m"].mean() > 200:
            df["t2m"] = df["t2m"] - 273.15

        # Rename temperature column to city name for merge clarity
        col_name = city_name
        df = df.rename(columns={"t2m": col_name})
        city_names.append(col_name)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="valid_time", how="inner")

    # --- TEMPORAL ALIGNMENT (CRITICAL) ---
    # Sort chronologically to guarantee window consistency
    merged_df = merged_df.sort_values("valid_time").reset_index(drop=True)

    # Drop any rows with missing values after alignment
    merged_df = merged_df.dropna().reset_index(drop=True)

    timestamps = pd.to_datetime(merged_df["valid_time"])
    data_matrix = merged_df[city_names].values.astype(np.float64)

    return data_matrix, city_names, timestamps


def validate_data(data_matrix, city_names):
    """
    Print dataset shape, basic stats per sensor, and verify no NaN values.
    Raises ValueError if validation fails.
    """
    time_steps, num_sensors = data_matrix.shape

    print(f"\n{'='*60}")
    print(f"  REAL MULTI-SENSOR DATA VALIDATION")
    print(f"{'='*60}")
    print(f"  Shape: ({time_steps}, {num_sensors})  [time_steps x sensors]")
    print(f"  Sensors: {num_sensors}")
    print(f"  Time steps: {time_steps}")
    print()

    nan_count = np.isnan(data_matrix).sum()
    if nan_count > 0:
        raise ValueError(f"Data contains {nan_count} NaN values! Aborting.")
    print(f"  NaN check: PASSED (0 NaN values)")
    print()

    print(f"  {'City':<20} {'Min (°C)':>10} {'Max (°C)':>10} {'Mean (°C)':>10} {'Std (°C)':>10}")
    print(f"  {'-'*60}")

    for i, name in enumerate(city_names):
        col = data_matrix[:, i]
        print(f"  {name:<20} {col.min():>10.2f} {col.max():>10.2f} {col.mean():>10.2f} {col.std():>10.2f}")

    print(f"{'='*60}\n")

    return True
