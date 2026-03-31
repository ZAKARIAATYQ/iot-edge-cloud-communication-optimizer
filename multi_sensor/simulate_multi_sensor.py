import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import csv

# Add parent directory to path to import from original project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate_iot_system import GRUModel
from multi_sensor.multi_sensor_utils import generate_multi_sensor_data
from multi_sensor.multi_reconstruction import reconstruct_multi_sensor_cloud, decide_transmissions_multi
from multi_sensor.multi_metrics import (
    compute_multi_metrics, save_multi_metrics_summary, save_detailed_metrics,
    plot_multi_sensor_samples, plot_error_distribution_across_sensors, plot_edge_vs_reconstructed_validation
)

# --- CONFIGURATION ---
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sensor_dataset.csv")
MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "window72.pth")
WINDOW_SIZE = 72

SENSOR_COUNTS = [10, 50, 100]
EPSILONS = [0.5, 1.0]

def run_multi_sensor_scenario(scenario_name, epsilon, num_sensors, data_celsius, model, scaler, base_dir):
    print(f"  -> Running {scenario_name} (Sensors: {num_sensors}, Epsilon: {epsilon})")
    
    start_time = time.time()
    
    total_data = len(data_celsius)
    transmitted_points = 0
    
    # Pre-allocate arrays for speed and independent tracking
    y_true = np.zeros((total_data, num_sensors))
    y_reconstructed = np.zeros((total_data, num_sensors))
    edge_predictions_history = np.zeros((total_data, num_sensors))
    
    # Store the actual values
    y_true[:] = data_celsius[:]
    
    # Initial warm-up phase (t < WINDOW_SIZE)
    # Assumes all sensors transmit baseline data to establish identical history
    for t in range(WINDOW_SIZE):
        y_reconstructed[t, :] = y_true[t, :]
        transmitted_points += num_sensors
        edge_predictions_history[t, :] = y_true[t, :] # No real prediction in warmup
        
    # Main simulation loop
    # For performance and to prevent shared state cross-contamination, we use batched inference
    # Each sensor's window is purely extracted from its own column of historical data.
    
    for t in range(WINDOW_SIZE, total_data):
        # 1. Edge & Cloud identical window extraction
        # Both Edge and Cloud use the exact same historical Reconstructed data to ensure they stay in sync.
        # This is standard in Dual Prediction Schemes (DPS). By simulating the Cloud's state,
        # the Edge knows exactly what the Cloud will predict without asking.
        
        # Shape: (WINDOW_SIZE, num_sensors)
        window = y_reconstructed[t - WINDOW_SIZE : t, :]
        
        # 2. Scale data 
        # Scaler expects (samples, 1), so we flatten, transform, and reshape back.
        window_flat = window.reshape(-1, 1)
        window_scaled = scaler.transform(window_flat)
        window_scaled = window_scaled.reshape(WINDOW_SIZE, num_sensors)
        
        # Prepare tensor exactly matching (batch_size, seq_len, input_size) => (num_sensors, WINDOW_SIZE, 1)
        # We transpose the window to get (num_sensors, WINDOW_SIZE)
        window_scaled_t = window_scaled.T
        x_tensor = torch.tensor(window_scaled_t, dtype=torch.float32).unsqueeze(-1)
        
        # 3. Identical GRU Inference (Batch prediction for all sensors concurrently)
        with torch.no_grad():
            pred_scaled = model(x_tensor) # Output shape: (num_sensors, 1)
            
        # Inverse transform predictions
        pred_scaled_np = pred_scaled.numpy()
        pred_temps = scaler.inverse_transform(pred_scaled_np).flatten() # Shape: (num_sensors,)
        edge_predictions_history[t, :] = pred_temps
        
        # 4. Independent Transmission Logic per sensor
        real_temps = y_true[t, :]
        transmits = decide_transmissions_multi(real_temps, pred_temps, epsilon)
        
        transmitted_points += np.sum(transmits)
        
        # 5. Independent Cloud Reconstruction per sensor
        recon = reconstruct_multi_sensor_cloud(pred_temps, real_temps, transmits)
        y_reconstructed[t, :] = recon

    execution_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_multi_metrics(y_true, y_reconstructed, total_data, transmitted_points, num_sensors)
    
    # Ensure directory exists
    result_dir = os.path.join(base_dir, f"sensors_{num_sensors}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save results
    save_multi_metrics_summary(result_dir, scenario_name, epsilon, metrics, execution_time)
    
    # Create scenario specific folder for detailed artifacts
    scenario_dir = os.path.join(result_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)
    
    save_detailed_metrics(scenario_dir, metrics)
    plot_multi_sensor_samples(y_true, y_reconstructed, scenario_dir)
    plot_error_distribution_across_sensors(metrics, scenario_dir)
    plot_edge_vs_reconstructed_validation(y_true, y_reconstructed, edge_predictions_history, scenario_dir)
    
    print(f"    Finished. Time: {execution_time:.2f}s, Reduction: {metrics['reduction']:.2f}%")
    return metrics


def main():
    print("Initializing Multi-Sensor IoT-Edge-Cloud Simulation...")
    
    # 1. Load Original Dataset
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return
        
    temp_col = 't2m' if 't2m' in df.columns else (
        'temperature' if 'temperature' in df.columns else df.select_dtypes(include=[np.number]).columns[-1]
    )
    temp_kelvin = df[temp_col].values
    base_celsius = temp_kelvin - 273.15 if np.mean(temp_kelvin) > 200 else temp_kelvin
    print(f"Dataset loaded. Length: {len(base_celsius)} points.")
    
    # Fits scaler on base data
    scaler = MinMaxScaler()
    scaler.fit(base_celsius.reshape(-1, 1))
    
    # Load identical GRU Model
    model = GRUModel(input_size=1, hidden_size=64, num_layers=1)
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        model.eval()
        print("Identical GRU model parameters loaded for Edge/Cloud.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(base_dir, exist_ok=True)
    
    # 2. Main Experimental Loop
    for num_sensors in SENSOR_COUNTS:
        print(f"\n--- Starting Evaluation for {num_sensors} Sensors ---")
        
        # Generate independent sensor data mimicking the real world
        multi_data = generate_multi_sensor_data(base_celsius, num_sensors)
        
        for epsilon in EPSILONS:
            scenario_name = f"epsilon_{str(epsilon).replace('.', '_')}"
            run_multi_sensor_scenario(
                scenario_name=scenario_name,
                epsilon=epsilon,
                num_sensors=num_sensors,
                data_celsius=multi_data,
                model=model,
                scaler=scaler,
                base_dir=base_dir
            )
            
    print("\nAll Multi-Sensor Simulations Completed Successfully.")

if __name__ == "__main__":
    main()
