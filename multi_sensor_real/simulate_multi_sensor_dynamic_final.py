"""
simulate_multi_sensor_dynamic_final.py
---------------------------------
Dynamic-threshold variant of the REAL multi-sensor IoT-Edge-Cloud simulation
using FIXED optimal hyperparameters.

Replaces the fixed epsilon with a per-sensor **dynamic threshold** that adapts
based on recent MAE, real-signal variance, and remaining energy:

    epsilon(t) = α·(1 - MAE_norm) + β·(1 - Var_norm) + γ·(1 - Energy_norm)

Key design rules:
    • Does NOT modify any existing file or function.
    • Reuses the EXACT same GRU model, data pipeline, metrics, and plot
      utilities already defined in the project.
    • Output structure is identical to the fixed-epsilon scenarios so that
      results are directly comparable.

Run with:
    python -m multi_sensor_real.simulate_multi_sensor_dynamic_final
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

# --- Reused imports (NO reimplementation) ---
from simulate_iot_system import GRUModel
from multi_sensor.multi_reconstruction import reconstruct_multi_sensor_cloud
from multi_sensor.multi_metrics import (
    compute_multi_metrics,
    save_multi_metrics_summary,
    save_detailed_metrics,
    plot_multi_sensor_samples,
    plot_error_distribution_across_sensors,
    plot_edge_vs_reconstructed_validation,
)
from multi_sensor_real.multi_real_utils import (
    load_real_multi_sensor_data,
    validate_data,
)
from multi_sensor_real.simulate_multi_sensor_real import (
    save_per_sensor_metrics_csv,
    detect_critical_events,
    plot_transmission_per_city,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FILE = os.path.join(_PROJECT_ROOT, "model", "window72.pth")
WINDOW_SIZE = 72
NUM_SENSORS = 5
CRITICAL_THRESHOLD = 2.0

# Dynamic threshold hyper-parameters (FIXED OPTIMAL VALUES)
ALPHA = 0.5
BETA = 0.4
GAMMA = 0.1
LOOKBACK = 10          # number of recent steps for MAE / variance

# Energy simulation constants
TX_COST = 0.005         # energy cost per transmission
COMP_COST = 0.0005      # energy cost per computation step (always applied)
SOLAR_CHARGE_RATE = 0.02 # solar charging rate

# Normalization ceilings (used to map raw values to [0, 1])
MAE_MAX = 5.0          # °C — practical upper bound for recent MAE
VAR_MAX = 25.0         # °C² — practical upper bound for recent variance


# EMA state for per-sensor smoothed MAE (module-level, lightweight)
_ema_mae = {}  # key: sensor_idx -> smoothed MAE value
EMA_ALPHA = 0.8  # weight for previous smoothed value

# Hysteresis & dead zone constants
DEAD_ZONE = 0.3   # errors below this are suppressed to zero
HYST_DELTA = 0.05  # half-width of hysteresis band around epsilon

# Per-sensor transmission state (for hysteresis memory)
_tx_state = {}  # key: sensor_idx -> bool (previous transmit decision)


# ---------------------------------------------------------------------------
# Dynamic threshold computation
# ---------------------------------------------------------------------------
def compute_dynamic_epsilon(
    y_true, y_reconstructed, t, sensor_idx,
    energy,
    mae_max=MAE_MAX, var_max=VAR_MAX,
    alpha=ALPHA, beta=BETA, gamma=GAMMA,
    lookback=LOOKBACK,
):
    """
    Compute the dynamic threshold for one sensor at time step *t*.

    epsilon(t) = α·(1 - MAE_norm) + β·(1 - Var_norm) + γ·(1 - Energy_norm)

    Parameters
    ----------
    y_true : np.ndarray, shape (T, S)
    y_reconstructed : np.ndarray, shape (T, S)
    t : int — current time step (exclusive upper bound of look-back window)
    sensor_idx : int
    energy : float — current energy level for this sensor (0–100)
    """
    start = max(0, t - lookback)

    # --- MAE over recent reconstruction errors ---
    recent_real = y_true[start:t, sensor_idx]
    recent_recon = y_reconstructed[start:t, sensor_idx]
    current_mae = np.mean(np.abs(recent_real - recent_recon)) if len(recent_real) > 0 else 0.0

    # --- EMA smoothing (per sensor) ---
    if sensor_idx in _ema_mae:
        mae = EMA_ALPHA * _ema_mae[sensor_idx] + (1.0 - EMA_ALPHA) * current_mae
    else:
        mae = current_mae  # first step: initialise with raw MAE
    _ema_mae[sensor_idx] = mae

    mae_norm = min(mae / mae_max, 1.0)

    # --- Variance of the *real* signal (measures data volatility) ---
    variance = np.var(recent_real) if len(recent_real) > 1 else 0.0
    var_norm = min(variance / var_max, 1.0)

    # --- Energy normalisation ---
    energy_norm = min(max(energy / 100.0, 0.0), 1.0)

    # --- Composite dynamic epsilon (non-linear) ---
    eps = alpha * ((1.0 - mae_norm) ** 2) + beta * ((1.0 - var_norm) ** 2) + gamma * ((1.0 - energy_norm) ** 2)

    # --- Minimum epsilon floor (prevents over-sensitive thresholds) ---
    eps = max(eps, 0.3)

    return eps


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
# Dynamic-threshold scenario runner
# ---------------------------------------------------------------------------
def run_dynamic_threshold_scenario(
    scenario_name, data_celsius, city_names, timestamps,
    model, scaler, base_dir, alpha, beta, gamma
):
    """
    Run the dynamic-threshold scenario on real multi-sensor data using optimal weights.

    The loop mirrors run_real_multi_sensor_scenario() exactly, but replaces
    the fixed epsilon decision with per-sensor dynamic thresholds and tracks
    energy consumption.
    """
    num_sensors = data_celsius.shape[1]
    total_data = data_celsius.shape[0]

    print(f"\n  -> Running {scenario_name} (Sensors: {num_sensors}, Threshold: DYNAMIC)")
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
        
        # Energy cost during warm-up: transmit + compute
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

        # 3. Dynamic transmission decision — EDGE drives decision
        real_temps = y_true[t, :]
        transmits = np.zeros(num_sensors, dtype=int)

        for i in range(num_sensors):
            eps_dyn = compute_dynamic_epsilon(
                y_true, y_edge_history, t, i, edge_energy,
                alpha=alpha, beta=beta, gamma=gamma
            )

            # Energy-aware adaptation
            if edge_energy < 20:
                eps_dyn *= 1.2

            if edge_energy < 10:
                eps_dyn *= 1.4

            # Dead zone: suppress small errors (edge prediction only)
            error = abs(real_temps[i] - edge_pred_temps[i])
            if error < DEAD_ZONE:
                error = 0.0

            # Hysteresis decision
            eps_high = eps_dyn + HYST_DELTA
            eps_low  = eps_dyn - HYST_DELTA
            prev = _tx_state.get(i, False)

            if error > eps_high:
                transmits[i] = 1
                _tx_state[i] = True
            elif error < eps_low:
                transmits[i] = 0
                _tx_state[i] = False
            else:
                transmits[i] = int(prev)
                # _tx_state[i] unchanged

        transmitted_points += int(np.sum(transmits))
        per_sensor_tx += transmits

        # Computation contribution (all sensors)
        comp_energy_per_sensor += COMP_COST

        # Transmission contribution
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
        y_true, y_cloud_history, total_data, transmitted_points, num_sensors,
    )

    # --- Critical events detection (reuses existing function) ---
    total_critical, missed_critical = detect_critical_events(y_true, y_cloud_history)
    metrics["critical_events"] = total_critical
    metrics["missed_critical"] = missed_critical

    # --- Save results ---
    result_dir = os.path.join(base_dir, f"sensors_{num_sensors}")
    os.makedirs(result_dir, exist_ok=True)

    # Append to shared summary.csv (epsilon column = "dynamic")
    save_multi_metrics_summary(result_dir, scenario_name, "dynamic", metrics, execution_time)

    # Scenario sub-folder
    scenario_dir = os.path.join(result_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # Detailed metrics (reuses existing saver)
    save_detailed_metrics(scenario_dir, metrics)

    # Enhanced per-sensor CSV with city names (reuses existing function)
    save_per_sensor_metrics_csv(scenario_dir, metrics, city_names, per_sensor_tx)

    # Critical events info
    crit_path = os.path.join(scenario_dir, "critical_events.txt")
    with open(crit_path, "w", encoding="utf-8") as f:
        f.write(f"Critical Event Threshold: |ΔT| > {CRITICAL_THRESHOLD} °C\n")
        f.write(f"Total Critical Events: {total_critical}\n")
        f.write(f"Missed Critical Events: {missed_critical}\n")
        if total_critical > 0:
            f.write(f"Detection Rate: {(1 - missed_critical / total_critical) * 100:.2f}%\n")

    # Energy report
    save_energy_report(scenario_dir, city_names, edge_energy, comp_energy_per_sensor, tx_energy_per_sensor)

    # --- Plots (reuse existing + transmission chart) ---
    plot_multi_sensor_samples(y_true, y_cloud_history, scenario_dir)
    plot_error_distribution_across_sensors(metrics, scenario_dir)
    plot_edge_vs_reconstructed_validation(
        y_true, y_cloud_history, edge_predictions_history, scenario_dir,
    )
    plot_transmission_per_city(
        city_names, per_sensor_tx, total_data, scenario_dir,
    )

    print(f"    Finished. Time: {execution_time:.2f}s, "
          f"Reduction: {metrics['reduction']:.2f}%, "
          f"Critical Events: {total_critical} (missed: {missed_critical})")

    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)
    print("=" * 70)
    print("  REAL Multi-Sensor IoT-Edge-Cloud — DYNAMIC THRESHOLD Simulation")
    print("  Using 5 Moroccan City Datasets as Independent Sensors")
    print("=" * 70)

    # 1. Load & validate real sensor data
    print("\n[1/4] Loading real sensor data...")
    data_matrix, city_names, timestamps = load_real_multi_sensor_data()
    validate_data(data_matrix, city_names)

    # 2. Fit scaler on FULL combined data (global min/max across all sensors)
    print("[2/4] Fitting global MinMaxScaler on combined data...")
    scaler = MinMaxScaler()
    scaler.fit(data_matrix.reshape(-1, 1))
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

    # Use results2 for the final clean output
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results2")
    os.makedirs(base_dir, exist_ok=True)

    # Reset EMA and hysteresis state before full simulation
    _ema_mae.clear()
    _tx_state.clear()
    
    # 4. Run the full dynamic threshold scenario with optimal fixed weights
    print(f"\n[4/4] Running FULL DYNAMIC THRESHOLD scenario with optimal weights...")
    metrics = run_dynamic_threshold_scenario(
        scenario_name="dynamic_threshold",
        data_celsius=data_matrix,
        city_names=city_names,
        timestamps=timestamps,
        model=model,
        scaler=scaler,
        base_dir=base_dir,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA
    )

    # Final summary
    print("\n" + "=" * 70)
    print("  DYNAMIC THRESHOLD SIMULATION COMPLETE — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Reduction: {metrics['reduction']:.2f}%  |  "
          f"MAE: {metrics['global_mae']:.4f}  |  "
          f"RMSE: {metrics['global_rmse']:.4f}  |  "
          f"R²: {metrics['global_r2']:.4f}")
    print(f"  Critical Events: {metrics['critical_events']}  |  "
          f"Missed: {metrics['missed_critical']}")
    print("=" * 70)
    print(f"\nResults saved to: {base_dir}")


if __name__ == "__main__":
    main()
