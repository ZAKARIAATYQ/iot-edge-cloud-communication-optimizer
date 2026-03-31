import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import csv

from filtering_logic import decide_transmission
from reconstruction import reconstruct_cloud_data
from metrics import (
    compute_metrics, analyze_critical_events, save_metrics,
    plot_real_vs_reconstructed, plot_error_boxplot, plot_residuals
)

# --- CONFIGURATION ---
DATA_FILE = "sensor_dataset.csv"
MODEL_FILE = "model/window72.pth"
WINDOW_SIZE = 72

SCENARIOS = [
    {"name": "baseline", "epsilon": None, "dir": "results/baseline"},
    {"name": "epsilon_0_5", "epsilon": 0.5, "dir": "results/epsilon_0_5"},
    {"name": "epsilon_1_0", "epsilon": 1.0, "dir": "results/epsilon_1_0"}
]
SUMMARY_FILE = "results/simulation_summary.csv"

# --- MODEL DEFINITION ---
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def run_simulation_scenario(scenario, data_celsius, model, scaler):
    print(f"Running simulation for scenario: {scenario['name']} (Epsilon: {scenario['epsilon']})")
    
    y_true = []
    y_reconstructed = []
    transmission_flags = []
    
    total_data = len(data_celsius)
    transmitted_points = 0
    
    for t in range(total_data):
        real_temp = data_celsius[t]
        y_true.append(real_temp)
        
        if t < WINDOW_SIZE:
            # Warm-up
            y_reconstructed.append(real_temp)
            transmission_flags.append(True)
            transmitted_points += 1
            continue
            
        # 🔥 MODIFICA QUI: uso valori ricostruiti
        window = np.array(y_reconstructed[t - WINDOW_SIZE : t])
        
        # Scale input
        window_scaled = scaler.transform(window.reshape(-1, 1))
        
        # Prepare tensor
        x_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            pred_scaled = model(x_tensor).item()
            
        pred_temp = scaler.inverse_transform([[pred_scaled]])[0][0]
        
        # Filtering
        transmit = decide_transmission(real_temp, pred_temp, scenario['epsilon'])
        transmission_flags.append(transmit)
        
        if transmit:
            transmitted_points += 1
            
        # Reconstruction
        reconstructed_value = reconstruct_cloud_data(real_temp, pred_temp, transmit)
        y_reconstructed.append(reconstructed_value)

    # Metrics
    metrics = compute_metrics(y_true, y_reconstructed, total_data, transmitted_points)
    crit_events, missed_events = analyze_critical_events(y_true, y_reconstructed, transmission_flags)
    
    result_dir = scenario["dir"]
    os.makedirs(result_dir, exist_ok=True)
    
    save_metrics(result_dir, metrics, crit_events, missed_events)
    plot_real_vs_reconstructed(y_true, y_reconstructed, result_dir)
    plot_error_boxplot(y_true, y_reconstructed, result_dir)
    plot_residuals(y_true, y_reconstructed, result_dir)
    
    print(f"Scenario {scenario['name']} finished. Transmission Reduction: {metrics['reduction']:.2f}%")
    return metrics


def main():
    print("Initializing IoT-Edge-Cloud Simulation...")
    
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return
        
    print(f"Dataset loaded. Number of rows: {len(df)}")
    
    if 't2m' in df.columns:
        temp_col = 't2m'
    elif 'temperature' in df.columns:
        temp_col = 'temperature'
    else:
        temp_col = df.select_dtypes(include=[np.number]).columns[-1]
    
    print(f"Using column '{temp_col}' for temperature data.")
    
    temp_kelvin = df[temp_col].values
    
    if np.mean(temp_kelvin) > 200:
        temp_celsius = temp_kelvin - 273.15
    else:
        temp_celsius = temp_kelvin
        
    scaler = MinMaxScaler()
    scaler.fit(temp_celsius.reshape(-1, 1))
    
    model = GRUModel(input_size=1, hidden_size=64, num_layers=1)
    # Apply dynamic quantization to match the saved quantized model's architecture
    model = torch.quantization.quantize_dynamic(
        model, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )
    try:
        # weights_only=False required for quantized models (PyTorch 2.6+)
        model.load_state_dict(
            torch.load(MODEL_FILE, map_location=torch.device('cpu'), weights_only=False)
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    with open(SUMMARY_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'epsilon', 'transmitted', 'reduction_percentage', 'MAE', 'RMSE', 'R2'])
    
    for scenario in SCENARIOS:
        metrics = run_simulation_scenario(scenario, temp_celsius, model, scaler)
        
        with open(SUMMARY_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                scenario['name'],
                scenario['epsilon'] if scenario['epsilon'] is not None else 'N/A',
                metrics['transmitted'],
                round(metrics['reduction'], 2),
                round(metrics['mae'], 4),
                round(metrics['rmse'], 4),
                round(metrics['r2'], 4)
            ])
            
    print("Simulation complete.")


if __name__ == "__main__":
    main()