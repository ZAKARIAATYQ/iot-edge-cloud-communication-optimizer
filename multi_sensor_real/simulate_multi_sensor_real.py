"""
simulate_multi_sensor_real.py
-----------------------------
Main simulation driver for the REAL multi-sensor IoT-Edge-Cloud experiment.

Uses 5 geographically distinct Moroccan city datasets instead of synthetic
noise-based sensors.  Reuses the EXACT same GRU model, reconstruction logic,
and metrics from the existing multi-sensor module — no reimplementation.

This module does NOT modify any existing files or modules.
"""

import os
import sys
import time
import csv
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Imports from existing modules (NO reimplementation) ---
from simulate_iot_system import GRUModel
from multi_sensor.multi_reconstruction import (
    reconstruct_multi_sensor_cloud,
    decide_transmissions_multi,
)
from multi_sensor.multi_metrics import (
    compute_multi_metrics,
    save_multi_metrics_summary,
    save_detailed_metrics,
    plot_multi_sensor_samples,
    plot_error_distribution_across_sensors,
    plot_edge_vs_reconstructed_validation,
)

# --- Import from our new utils ---
from multi_sensor_real.multi_real_utils import (
    load_real_multi_sensor_data,
    validate_data,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FILE = os.path.join(_PROJECT_ROOT, "model", "window72.pth")
WINDOW_SIZE = 72
NUM_SENSORS = 5
EPSILONS = [0.5, 1.0]

# Critical event threshold (|ΔT| > 2 °C between consecutive steps)
CRITICAL_THRESHOLD = 2.0

# Energy simulation constants
TX_COST = 0.005         # energy cost per transmission
COMP_COST = 0.0005      # energy cost per computation step (always applied)
SOLAR_CHARGE_RATE = 0.02 # solar charging rate

# ---------------------------------------------------------------------------
# Energy report saver
# ---------------------------------------------------------------------------
def save_energy_report(result_dir, city_names, edge_energy, comp_energy, tx_energy):
    """
    Save a human-readable energy report using the shared edge energy model.
    """
    total_consumed = np.sum(comp_energy) + np.sum(tx_energy)
    filepath = os.path.join(result_dir, "energy_report.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("--- EDGE ENERGY REPORT ---\n\n")
        f.write(f"Initial Energy: Random Uniform [80, 100]%\n")
        f.write(f"Final Energy: {edge_energy:.4f}%\n")
        f.write(f"Total Consumed: {total_consumed:.4f}\n")
        f.write(f"Solar Charge Rate: {SOLAR_CHARGE_RATE}\n\n")
        f.write(f"Per Sensor Energy Usage:\n")
        f.write(f"{'City':<20} {'Compute':>15} {'Transmission':>15} {'Total':>15}\n")
        f.write(f"{'-'*68}\n")
        for i, name in enumerate(city_names):
            total_s = comp_energy[i] + tx_energy[i]
            f.write(f"{name:<20} {comp_energy[i]:>15.4f} {tx_energy[i]:>15.4f} {total_s:>15.4f}\n")

# ---------------------------------------------------------------------------
# Enhanced per-sensor metrics with city names & transmission counts
# ---------------------------------------------------------------------------
def save_per_sensor_metrics_csv(result_dir, metrics, city_names, per_sensor_tx):
    """
    Save per_sensor_metrics.csv with city names, MAE, RMSE, R², and
    per-sensor transmission counts.
    """
    filepath = os.path.join(result_dir, "per_sensor_metrics.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sensor_id", "city", "MAE", "RMSE", "R2",
            "transmissions", "total_points", "reduction_pct",
        ])
        total_pts = metrics["total"] // len(city_names)
        for i, city in enumerate(city_names):
            tx = int(per_sensor_tx[i])
            red = (1 - tx / total_pts) * 100 if total_pts > 0 else 0
            writer.writerow([
                i,
                city,
                round(metrics["per_sensor_mae"][i], 4),
                round(metrics["per_sensor_rmse"][i], 4),
                round(metrics["per_sensor_r2"][i], 4),
                tx,
                total_pts,
                round(red, 2),
            ])


# ---------------------------------------------------------------------------
# Critical events detection (consistent with single-sensor logic)
# ---------------------------------------------------------------------------
def detect_critical_events(y_true, y_reconstructed, threshold=CRITICAL_THRESHOLD):
    """
    A critical event is defined as a time step where the absolute change in
    the real signal exceeds `threshold` °C compared to the previous step.

    Returns:
        total_critical (int): number of critical events across all sensors.
        missed_critical (int): critical events where the reconstructed signal
            did not faithfully track the real value (error > threshold/2).
    """
    # Differences along time axis
    real_diff = np.abs(np.diff(y_true, axis=0))            # (T-1, S)
    recon_error = np.abs(y_true[1:] - y_reconstructed[1:]) # (T-1, S)

    critical_mask = real_diff > threshold
    total_critical = int(critical_mask.sum())

    # A critical event is "missed" when the reconstruction error during that
    # event exceeds half the threshold
    missed_mask = critical_mask & (recon_error > threshold / 2)
    missed_critical = int(missed_mask.sum())

    return total_critical, missed_critical


# ---------------------------------------------------------------------------
# Transmission-per-city bar chart
# ---------------------------------------------------------------------------
def plot_transmission_per_city(city_names, per_sensor_tx, total_pts_per_sensor, result_dir):
    """
    Bar chart showing transmissions vs total points for each city.
    """
    x = np.arange(len(city_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_total = ax.bar(x - width / 2, [total_pts_per_sensor] * len(city_names),
                        width, label="Total Points", color="#4c72b0", alpha=0.7)
    bars_tx = ax.bar(x + width / 2, per_sensor_tx,
                     width, label="Transmitted", color="#dd8452", alpha=0.9)

    ax.set_xlabel("City (Sensor)")
    ax.set_ylabel("Number of Points")
    ax.set_title("Transmission Statistics per City")
    ax.set_xticks(x)
    ax.set_xticklabels(city_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add reduction % labels on top of transmitted bars
    for i, (tx, bar) in enumerate(zip(per_sensor_tx, bars_tx)):
        red = (1 - tx / total_pts_per_sensor) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"-{red:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "transmission_per_city.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main simulation scenario runner
# ---------------------------------------------------------------------------
def run_real_multi_sensor_scenario(
    scenario_name, epsilon, data_celsius, city_names, timestamps,
    model, scaler, base_dir
):
    """
    Run one scenario (one epsilon value) on the real multi-sensor data.
    Mirrors the logic of simulate_multi_sensor.run_multi_sensor_scenario
    exactly, but adapted for real data with city labels.
    """
    num_sensors = data_celsius.shape[1]
    total_data = data_celsius.shape[0]

    print(f"\n  -> Running {scenario_name} (Sensors: {num_sensors}, Epsilon: {epsilon})")
    start_time = time.time()

    # Pre-allocate tracking arrays — INDEPENDENT edge and cloud histories
    y_true = np.copy(data_celsius)
    y_edge_history = np.zeros_like(y_true)    # edge's own reconstructed signal
    y_cloud_history = np.zeros_like(y_true)   # cloud's own reconstructed signal
    edge_predictions_history = np.zeros_like(y_true)
    per_sensor_tx = np.zeros(num_sensors, dtype=int)
    
    edge_energy = float(np.random.uniform(80, 100))
    comp_energy_per_sensor = np.zeros(num_sensors)
    tx_energy_per_sensor = np.zeros(num_sensors)

    transmitted_points = 0

    # --- Warm-up phase (t < WINDOW_SIZE): all sensors transmit ---
    for t in range(WINDOW_SIZE):
        y_edge_history[t, :] = y_true[t, :]
        y_cloud_history[t, :] = y_true[t, :]
        edge_predictions_history[t, :] = y_true[t, :]
        transmitted_points += num_sensors
        per_sensor_tx += 1
        
        edge_energy -= num_sensors * COMP_COST
        edge_energy -= num_sensors * TX_COST
        
        hour = timestamps[t].hour
        solar_factor = max(0, math.sin((hour - 6) / 12 * math.pi))
        edge_energy += SOLAR_CHARGE_RATE * solar_factor
        
        edge_energy = max(0.0, min(edge_energy, 100.0))
        
        comp_energy_per_sensor += COMP_COST
        tx_energy_per_sensor += TX_COST


    # --- Main simulation loop ---
    for t in range(WINDOW_SIZE, total_data):
        # 1a. EDGE sliding window (from edge's own history)
        edge_window = y_edge_history[t - WINDOW_SIZE: t, :]
        edge_flat = edge_window.reshape(-1, 1)
        edge_scaled = scaler.transform(edge_flat).reshape(WINDOW_SIZE, num_sensors)
        edge_tensor = torch.tensor(edge_scaled.T, dtype=torch.float32).unsqueeze(-1)

        # 1b. CLOUD sliding window (from cloud's own history)
        cloud_window = y_cloud_history[t - WINDOW_SIZE: t, :]
        cloud_flat = cloud_window.reshape(-1, 1)
        cloud_scaled = scaler.transform(cloud_flat).reshape(WINDOW_SIZE, num_sensors)
        cloud_tensor = torch.tensor(cloud_scaled.T, dtype=torch.float32).unsqueeze(-1)

        # 2. Independent GRU inference for edge and cloud
        with torch.no_grad():
            edge_pred_scaled = model(edge_tensor)    # (num_sensors, 1)
            cloud_pred_scaled = model(cloud_tensor)  # (num_sensors, 1)

        edge_pred_temps = scaler.inverse_transform(edge_pred_scaled.numpy()).flatten()
        cloud_pred_temps = scaler.inverse_transform(cloud_pred_scaled.numpy()).flatten()
        edge_predictions_history[t, :] = edge_pred_temps

        # 3. Transmission decision — EDGE drives decision
        real_temps = y_true[t, :]
        transmits = decide_transmissions_multi(real_temps, edge_pred_temps, epsilon)
        transmitted_points += int(np.sum(transmits))
        per_sensor_tx += transmits.astype(int)

        comp_energy_per_sensor += COMP_COST
        for i in range(num_sensors):
            if transmits[i]:
                tx_energy_per_sensor[i] += TX_COST

        edge_energy -= num_sensors * COMP_COST
        edge_energy -= np.sum(transmits) * TX_COST
        
        hour = timestamps[t].hour
        solar_factor = max(0, math.sin((hour - 6) / 12 * math.pi))
        edge_energy += SOLAR_CHARGE_RATE * solar_factor
        
        edge_energy = max(0.0, min(edge_energy, 100.0))

        # 4. Update INDEPENDENT histories per sensor
        for i in range(num_sensors):
            if transmits[i]:
                # Transmitted: both edge and cloud receive real value
                y_edge_history[t, i] = real_temps[i]
                y_cloud_history[t, i] = real_temps[i]
            else:
                # Not transmitted: each uses its OWN prediction
                y_edge_history[t, i] = edge_pred_temps[i]
                y_cloud_history[t, i] = cloud_pred_temps[i]

    execution_time = time.time() - start_time

    # --- Metrics — evaluated on CLOUD reconstruction (reuses existing function) ---
    metrics = compute_multi_metrics(
        y_true, y_cloud_history, total_data, transmitted_points, num_sensors
    )

    # --- Critical events detection ---
    total_critical, missed_critical = detect_critical_events(y_true, y_cloud_history)
    metrics["critical_events"] = total_critical
    metrics["missed_critical"] = missed_critical

    # --- Save results ---
    result_dir = os.path.join(base_dir, f"sensors_{num_sensors}")
    os.makedirs(result_dir, exist_ok=True)

    # Reuse existing summary saver (appends to sensors_5/summary.csv)
    save_multi_metrics_summary(result_dir, scenario_name, epsilon, metrics, execution_time)

    # Scenario sub-folder
    scenario_dir = os.path.join(result_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # Reuse existing detailed saver
    save_detailed_metrics(scenario_dir, metrics)

    # Enhanced per-sensor CSV with city names
    save_per_sensor_metrics_csv(scenario_dir, metrics, city_names, per_sensor_tx)

    # Save critical events info
    crit_path = os.path.join(scenario_dir, "critical_events.txt")
    with open(crit_path, "w", encoding="utf-8") as f:
        f.write(f"Critical Event Threshold: |ΔT| > {CRITICAL_THRESHOLD} °C\n")
        f.write(f"Total Critical Events: {total_critical}\n")
        f.write(f"Missed Critical Events: {missed_critical}\n")
        if total_critical > 0:
            f.write(f"Detection Rate: {(1 - missed_critical / total_critical) * 100:.2f}%\n")

    # Energy report
    save_energy_report(scenario_dir, city_names, edge_energy, comp_energy_per_sensor, tx_energy_per_sensor)

    # --- Plots (reuse existing + new transmission chart) ---
    plot_multi_sensor_samples(y_true, y_cloud_history, scenario_dir)
    plot_error_distribution_across_sensors(metrics, scenario_dir)
    plot_edge_vs_reconstructed_validation(
        y_true, y_cloud_history, edge_predictions_history, scenario_dir
    )
    plot_transmission_per_city(
        city_names, per_sensor_tx, total_data, scenario_dir
    )

    print(f"    Finished. Time: {execution_time:.2f}s, "
          f"Reduction: {metrics['reduction']:.2f}%, "
          f"Critical Events: {total_critical} (missed: {missed_critical})")

    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  REAL Multi-Sensor IoT-Edge-Cloud Predictive Simulation")
    print("  Using 5 Moroccan City Datasets as Independent Sensors")
    print("=" * 70)

    # 1. Load & validate real sensor data
    print("\n[1/4] Loading real sensor data...")
    data_matrix, city_names, timestamps = load_real_multi_sensor_data()
    validate_data(data_matrix, city_names)

    # 2. Fit scaler on FULL combined data (global min/max across all sensors)
    print("[2/4] Fitting global MinMaxScaler on combined data...")
    scaler = MinMaxScaler()
    scaler.fit(data_matrix.reshape(-1, 1))  # flatten all sensors together
    print(f"  Scaler range: [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}] °C")

    # 3. Load pre-trained GRU model (same as single/multi sensor)
    print("[3/4] Loading pre-trained GRU model...")
    # Step A: Instantiate the base unquantized architecture
    model = GRUModel(input_size=1, hidden_size=64, num_layers=1)
    
    # Step B: Apply dynamic quantization to match the saved model's architecture
    model = torch.quantization.quantize_dynamic(
        model, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )

    try:
        # Step C: Load state_dict specifying weights_only=False.
        # PyTorch 2.6 requires this since quantized weights use custom C++ Objects (torch.ScriptObject)
        model.load_state_dict(
            torch.load(MODEL_FILE, map_location=torch.device("cpu"), weights_only=False)
        )
        model.eval()
        print("  GRU model loaded successfully (identical Edge/Cloud weights).")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return

    # 4. Run scenarios
    print("\n[4/4] Running simulation scenarios...")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(base_dir, exist_ok=True)

    all_results = {}
    for epsilon in EPSILONS:
        scenario_name = f"epsilon_{str(epsilon).replace('.', '_')}"
        metrics = run_real_multi_sensor_scenario(
            scenario_name=scenario_name,
            epsilon=epsilon,
            data_celsius=data_matrix,
            city_names=city_names,
            timestamps=timestamps,
            model=model,
            scaler=scaler,
            base_dir=base_dir,
        )
        all_results[scenario_name] = metrics

    # Final summary
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE — RESULTS SUMMARY")
    print("=" * 70)
    for name, m in all_results.items():
        print(f"  {name}:")
        print(f"    Reduction: {m['reduction']:.2f}%  |  "
              f"MAE: {m['global_mae']:.4f}  |  "
              f"RMSE: {m['global_rmse']:.4f}  |  "
              f"R²: {m['global_r2']:.4f}")
        print(f"    Critical Events: {m['critical_events']}  |  "
              f"Missed: {m['missed_critical']}")
    print("=" * 70)
    print(f"\nResults saved to: {base_dir}")


if __name__ == "__main__":
    main()
