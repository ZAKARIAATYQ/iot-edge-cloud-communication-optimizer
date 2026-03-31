"""
simulate_multi_sensor_dynamic.py
---------------------------------
Dynamic-threshold variant of the REAL multi-sensor IoT-Edge-Cloud simulation.

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
    python -m multi_sensor_real.simulate_multi_sensor_dynamic
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

# Dynamic threshold hyper-parameters
ALPHA = 0.4
BETA = 0.3
GAMMA = 0.3
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
def evaluate_grid_combination(
    alpha, beta, gamma,
    data_celsius, timestamps, model, scaler, max_steps=1000
):
    """
    Run a short simulation to evaluate a specific (α, β, γ) combination.
    Returns the score: MAE + (1 - reduction_ratio).
    """
    num_sensors = data_celsius.shape[1]
    # Restrict to max_steps or total data
    total_data = min(data_celsius.shape[0], max_steps)

    y_true = np.copy(data_celsius[:total_data])
    y_reconstructed = np.zeros_like(y_true)
    edge_energy = float(np.random.uniform(80, 100))
    
    comp_energy_per_sensor = np.zeros(num_sensors)
    tx_energy_per_sensor = np.zeros(num_sensors)

    transmitted_points = 0

    # Warm-up phase
    warmup = min(WINDOW_SIZE, total_data)
    for t in range(warmup):
        y_reconstructed[t, :] = y_true[t, :]
        transmitted_points += num_sensors
        
        edge_energy -= num_sensors * COMP_COST
        edge_energy -= num_sensors * TX_COST
        
        hour = timestamps[t].hour
        solar_factor = max(0, math.sin((hour - 6) / 12 * math.pi))
        edge_energy += SOLAR_CHARGE_RATE * solar_factor
        
        edge_energy = max(0.0, min(edge_energy, 100.0))
        
        comp_energy_per_sensor += COMP_COST
        tx_energy_per_sensor += TX_COST

    # Main simulation loop
    for t in range(warmup, total_data):
        window = y_reconstructed[t - WINDOW_SIZE: t, :]
        window_flat = window.reshape(-1, 1)
        window_scaled = scaler.transform(window_flat)
        window_scaled = window_scaled.reshape(WINDOW_SIZE, num_sensors)
        window_scaled_t = window_scaled.T
        x_tensor = torch.tensor(window_scaled_t, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            pred_scaled = model(x_tensor)

        pred_temps = scaler.inverse_transform(pred_scaled.numpy()).flatten()
        real_temps = y_true[t, :]
        transmits = np.zeros(num_sensors, dtype=int)

        for i in range(num_sensors):
            eps_dyn = compute_dynamic_epsilon(
                y_true, y_reconstructed, t, i, edge_energy,
                alpha=alpha, beta=beta, gamma=gamma
            )

            # Energy-aware adaptation
            if edge_energy < 20:
                eps_dyn *= 1.2

            if edge_energy < 10:
                eps_dyn *= 1.4

            # Dead zone: suppress small errors
            error = abs(real_temps[i] - pred_temps[i])
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

        recon = reconstruct_multi_sensor_cloud(
            pred_temps, real_temps, transmits.astype(bool),
        )
        y_reconstructed[t, :] = recon

    total_points_all = (total_data - warmup) * num_sensors if total_data > warmup else 1
    # Avoid counting warmup transmissions in reduction to get a clearer picture of the filter
    reduction_ratio = 1.0 - ((transmitted_points - (warmup * num_sensors)) / total_points_all)
    
    # Calculate MAE only on the predicted part
    if total_data > warmup:
        mae = np.mean(np.abs(y_true[warmup:total_data] - y_reconstructed[warmup:total_data]))
    else:
        mae = 0.0

    score = mae + (1.0 - reduction_ratio)
    return score, mae, reduction_ratio * 100.0


def evaluate_lookback_hyst_combination(
    lookback_val, hyst_delta_val,
    alpha, beta, gamma,
    data_celsius, timestamps, model, scaler, max_steps=2000
):
    """
    Run a short simulation to evaluate a specific (LOOKBACK, HYST_DELTA)
    combination.  Returns MAE, reduction%, missed_critical, total_critical,
    and a composite score.
    """
    # Reset global state for reproducibility
    _ema_mae.clear()
    _tx_state.clear()

    num_sensors = data_celsius.shape[1]
    total_data = min(data_celsius.shape[0], max_steps)

    y_true = np.copy(data_celsius[:total_data])
    y_reconstructed = np.zeros_like(y_true)
    edge_energy = float(np.random.uniform(80, 100))
    transmitted_points = 0

    warmup = min(WINDOW_SIZE, total_data)
    for t in range(warmup):
        y_reconstructed[t, :] = y_true[t, :]
        transmitted_points += num_sensors

        edge_energy -= num_sensors * COMP_COST
        edge_energy -= num_sensors * TX_COST
        hour = timestamps[t].hour
        solar_factor = max(0, math.sin((hour - 6) / 12 * math.pi))
        edge_energy += SOLAR_CHARGE_RATE * solar_factor
        edge_energy = max(0.0, min(edge_energy, 100.0))

    for t in range(warmup, total_data):
        window = y_reconstructed[t - WINDOW_SIZE: t, :]
        window_flat = window.reshape(-1, 1)
        window_scaled = scaler.transform(window_flat).reshape(WINDOW_SIZE, num_sensors)
        x_tensor = torch.tensor(window_scaled.T, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            pred_scaled = model(x_tensor)

        pred_temps = scaler.inverse_transform(pred_scaled.numpy()).flatten()
        real_temps = y_true[t, :]
        transmits = np.zeros(num_sensors, dtype=int)

        for i in range(num_sensors):
            eps_dyn = compute_dynamic_epsilon(
                y_true, y_reconstructed, t, i, edge_energy,
                alpha=alpha, beta=beta, gamma=gamma,
                lookback=lookback_val,
            )

            if edge_energy < 20:
                eps_dyn *= 1.2
            if edge_energy < 10:
                eps_dyn *= 1.4

            error = abs(real_temps[i] - pred_temps[i])
            if error < DEAD_ZONE:
                error = 0.0

            eps_high = eps_dyn + hyst_delta_val
            eps_low  = eps_dyn - hyst_delta_val
            prev = _tx_state.get(i, False)

            if error > eps_high:
                transmits[i] = 1
                _tx_state[i] = True
            elif error < eps_low:
                transmits[i] = 0
                _tx_state[i] = False
            else:
                transmits[i] = int(prev)

        transmitted_points += int(np.sum(transmits))

        edge_energy -= num_sensors * COMP_COST
        edge_energy -= np.sum(transmits) * TX_COST
        hour = timestamps[t].hour
        solar_factor = max(0, math.sin((hour - 6) / 12 * math.pi))
        edge_energy += SOLAR_CHARGE_RATE * solar_factor
        edge_energy = max(0.0, min(edge_energy, 100.0))

        recon = reconstruct_multi_sensor_cloud(
            pred_temps, real_temps, transmits.astype(bool),
        )
        y_reconstructed[t, :] = recon

    # --- Metrics ---
    total_points_all = (total_data - warmup) * num_sensors if total_data > warmup else 1
    reduction_ratio = 1.0 - ((transmitted_points - (warmup * num_sensors)) / total_points_all)

    if total_data > warmup:
        mae = np.mean(np.abs(y_true[warmup:total_data] - y_reconstructed[warmup:total_data]))
    else:
        mae = 0.0

    total_critical, missed_critical = detect_critical_events(
        y_true[:total_data], y_reconstructed[:total_data]
    )
    missed_rate = missed_critical / total_critical if total_critical > 0 else 0.0

    score = mae + (1.0 - reduction_ratio) + 0.5 * missed_rate
    return score, mae, reduction_ratio * 100.0, missed_critical, total_critical


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
    global LOOKBACK, HYST_DELTA
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

    # 4. Grid Search for optimal weights
    print("\n[4/5] Running GRID SEARCH for optimal weights (first 2000 steps)...")
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    betas  = [0.1, 0.2, 0.3, 0.4, 0.5]
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5]

    best_score = float('inf')
    best_weights = (0.4, 0.3, 0.3)  # default fallback
    best_metrics = (0.0, 0.0)

    for a in alphas:
        for b in betas:
            for g in gammas:
                # Constraint: sum to 1.0
                if abs(a + b + g - 1.0) > 1e-6:
                    continue
                
                score, mae, red = evaluate_grid_combination(
                    alpha=a, beta=b, gamma=g,
                    data_celsius=data_matrix, timestamps=timestamps, model=model, scaler=scaler,
                    max_steps=2000
                )
                
                if score < best_score:
                    best_score = score
                    best_weights = (a, b, g)
                    best_metrics = (mae, red)

    best_a, best_b, best_g = best_weights
    print(f"  Best alpha, beta, gamma: {best_a:.1f}, {best_b:.1f}, {best_g:.1f}")
    print(f"  Best score: {best_score:.4f} (MAE: {best_metrics[0]:.4f}, Reduction: {best_metrics[1]:.2f}%)")

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(base_dir, exist_ok=True)

    # Save best weights
    scenario_dir = os.path.join(base_dir, "sensors_5", "dynamic_threshold")
    os.makedirs(scenario_dir, exist_ok=True)
    weights_path = os.path.join(scenario_dir, "best_weights.txt")
    with open(weights_path, "w", encoding="utf-8") as f:
        f.write(f"Best alpha: {best_a:.1f}\n")
        f.write(f"Best beta:  {best_b:.1f}\n")
        f.write(f"Best gamma: {best_g:.1f}\n")
        f.write(f"Best score: {best_score:.4f}\n")

    # -----------------------------------------------------------------------
    # 4b. Grid Search for LOOKBACK and HYST_DELTA
    # -----------------------------------------------------------------------
    lookback_values = [5, 10, 12, 15, 20]
    hyst_values = [0.02, 0.05, 0.07, 0.1, 0.15]

    print(f"\n[4b/5] Running GRID SEARCH for LOOKBACK x HYST_DELTA "
          f"({len(lookback_values)}x{len(hyst_values)} = "
          f"{len(lookback_values) * len(hyst_values)} combos)...")

    lh_results = []
    best_lh_score = float('inf')
    best_lh = (LOOKBACK, HYST_DELTA)  # fallback
    best_lh_info = {}

    for lb in lookback_values:
        for hd in hyst_values:
            sc, mae_v, red_v, missed_v, total_v = evaluate_lookback_hyst_combination(
                lookback_val=lb, hyst_delta_val=hd,
                alpha=best_a, beta=best_b, gamma=best_g,
                data_celsius=data_matrix, timestamps=timestamps,
                model=model, scaler=scaler, max_steps=2000,
            )
            lh_results.append({
                'LOOKBACK': lb, 'HYST_DELTA': hd,
                'MAE': mae_v, 'Reduction': red_v,
                'Missed_Critical': missed_v, 'Total_Critical': total_v,
                'Score': sc,
            })
            print(f"    LOOKBACK={lb:>2}, HYST={hd:.2f} → "
                  f"MAE={mae_v:.4f}, Reduction={red_v:.2f}%, "
                  f"Missed={missed_v}, Score={sc:.4f}")

            if sc < best_lh_score:
                best_lh_score = sc
                best_lh = (lb, hd)
                best_lh_info = lh_results[-1]

    # Save CSV
    lh_csv_path = os.path.join(scenario_dir, "lookback_hyst_grid_search.csv")
    with open(lh_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'LOOKBACK', 'HYST_DELTA', 'MAE', 'Reduction',
            'Missed_Critical', 'Total_Critical', 'Score',
        ])
        writer.writeheader()
        for row in lh_results:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in row.items()})

    best_lb, best_hd = best_lh
    print(f"\n  Best LOOKBACK={best_lb}, HYST_DELTA={best_hd:.2f}")
    print(f"  Score={best_lh_score:.4f} (MAE={best_lh_info['MAE']:.4f}, "
          f"Reduction={best_lh_info['Reduction']:.2f}%, "
          f"Missed={best_lh_info['Missed_Critical']})")
    print(f"  Results saved to: {lh_csv_path}")

    # Apply best values globally for the full simulation
    LOOKBACK = best_lb
    HYST_DELTA = best_hd

    # Save best hyperparams
    with open(weights_path, "a", encoding="utf-8") as f:
        f.write(f"\nBest LOOKBACK:   {best_lb}\n")
        f.write(f"Best HYST_DELTA: {best_hd:.2f}\n")

    # Reset EMA and hysteresis state before full simulation
    _ema_mae.clear()
    _tx_state.clear()
    
    # 5. Run the full dynamic threshold scenario with optimal weights
    print(f"\n[5/5] Running FULL DYNAMIC THRESHOLD scenario with optimal weights...")
    metrics = run_dynamic_threshold_scenario(
        scenario_name="dynamic_threshold",
        data_celsius=data_matrix,
        city_names=city_names,
        timestamps=timestamps,
        model=model,
        scaler=scaler,
        base_dir=base_dir,
        alpha=best_a,
        beta=best_b,
        gamma=best_g
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
