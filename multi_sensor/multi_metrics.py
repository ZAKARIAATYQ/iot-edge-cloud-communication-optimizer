import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_multi_metrics(y_true, y_reconstructed, total_points_per_sensor, transmitted_points, num_sensors):
    """
    Computes communication and error metrics for a multi-sensor setup.
    y_true shape: (time_steps, num_sensors)
    y_reconstructed shape: (time_steps, num_sensors)
    """
    y_true = np.array(y_true)
    y_reconstructed = np.array(y_reconstructed)

    total_points_all = total_points_per_sensor * num_sensors
    reduction = (1 - transmitted_points / total_points_all) * 100 if total_points_all > 0 else 0
    
    # Calculate global metrics
    global_mae = mean_absolute_error(y_true, y_reconstructed)
    global_rmse = np.sqrt(mean_squared_error(y_true, y_reconstructed))
    global_r2 = r2_score(y_true, y_reconstructed)
    
    # Calculate per-sensor metrics
    per_sensor_mae = []
    per_sensor_rmse = []
    per_sensor_r2 = []
    
    for i in range(num_sensors):
        per_sensor_mae.append(mean_absolute_error(y_true[:, i], y_reconstructed[:, i]))
        per_sensor_rmse.append(np.sqrt(mean_squared_error(y_true[:, i], y_reconstructed[:, i])))
        per_sensor_r2.append(r2_score(y_true[:, i], y_reconstructed[:, i]))
        
    avg_ps_mae = np.mean(per_sensor_mae)
    avg_ps_rmse = np.mean(per_sensor_rmse)
    
    max_mae = np.max(per_sensor_mae)
    std_mae = np.std(per_sensor_mae)
    max_rmse = np.max(per_sensor_rmse)
    std_rmse = np.std(per_sensor_rmse)
    
    return {
        "total": total_points_all,
        "transmitted": transmitted_points,
        "reduction": reduction,
        "global_mae": global_mae,
        "global_rmse": global_rmse,
        "global_r2": global_r2,
        "avg_ps_mae": avg_ps_mae,
        "avg_ps_rmse": avg_ps_rmse,
        "max_mae": max_mae,
        "std_mae": std_mae,
        "max_rmse": max_rmse,
        "std_rmse": std_rmse,
        "per_sensor_mae": per_sensor_mae,
        "per_sensor_rmse": per_sensor_rmse,
        "per_sensor_r2": per_sensor_r2
    }

def save_multi_metrics_summary(result_dir, scenario_name, epsilon, metrics, time_taken=0.0):
    """
    Saves a summary.csv inside the scenario specific folder.
    metrics is a dict with all computed fields.
    """
    summary_file = os.path.join(result_dir, "summary.csv")
    
    write_header = not os.path.exists(summary_file)
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'scenario', 'epsilon', 'transmitted', 'reduction_percentage', 
                'global_MAE', 'global_RMSE', 'global_R2', 
                'max_MAE', 'std_MAE', 'max_RMSE', 'std_RMSE', 'execution_time_s'
            ])
        writer.writerow([
            scenario_name,
            epsilon if epsilon is not None else 'N/A',
            metrics['transmitted'],
            round(metrics['reduction'], 2),
            round(metrics['global_mae'], 4),
            round(metrics['global_rmse'], 4),
            round(metrics['global_r2'], 4),
            round(metrics['max_mae'], 4),
            round(metrics['std_mae'], 4),
            round(metrics['max_rmse'], 4),
            round(metrics['std_rmse'], 4),
            round(time_taken, 2)
        ])
        
def save_detailed_metrics(result_dir, metrics):
    """
    Saves detailed txt and per-sensor metrics
    """
    filepath = os.path.join(result_dir, "detailed_metrics.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("--- MULTI-SENSOR SIMULATION METRICS ---\n")
        f.write(f"Total Points Across All Sensors: {metrics['total']}\n")
        f.write(f"Transmitted Points: {metrics['transmitted']}\n")
        f.write(f"Transmission Reduction: {metrics['reduction']:.2f}%\n")
        f.write("\n")
        f.write("--- GLOBAL RECONSTRUCTION QUALITY ---\n")
        f.write(f"Global MAE: {metrics['global_mae']:.4f}\n")
        f.write(f"Global RMSE: {metrics['global_rmse']:.4f}\n")
        f.write(f"Global R2: {metrics['global_r2']:.4f}\n")
        f.write("\n")
        f.write("--- PER-SENSOR STATS ---\n")
        f.write(f"Average of Per-Sensor MAE: {metrics['avg_ps_mae']:.4f}\n")
        f.write(f"Max (Worst-Case) MAE: {metrics['max_mae']:.4f}\n")
        f.write(f"Std Dev of MAE Across Sensors: {metrics['std_mae']:.4f}\n")
        f.write(f"Max RMSE: {metrics['max_rmse']:.4f}\n")
        f.write(f"Std Dev of RMSE Across Sensors: {metrics['std_rmse']:.4f}\n")

    # Also save the per-sensor raw data
    per_sensor_filepath = os.path.join(result_dir, "per_sensor_errors.csv")
    with open(per_sensor_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sensor_id', 'MAE', 'RMSE', 'R2'])
        for i in range(len(metrics['per_sensor_mae'])):
             writer.writerow([
                 i,
                 round(metrics['per_sensor_mae'][i], 4),
                 round(metrics['per_sensor_rmse'][i], 4),
                 round(metrics['per_sensor_r2'][i], 4)
             ])

def plot_multi_sensor_samples(y_true, y_reconstructed, result_dir, num_samples=3):
    """
    Plot actual vs reconstructed for a few sample sensors.
    """
    num_sensors = y_true.shape[1]
    sensors_to_plot = np.random.choice(num_sensors, min(num_samples, num_sensors), replace=False)
    
    plt.figure(figsize=(15, 5 * len(sensors_to_plot)))
    
    for idx, s_idx in enumerate(sensors_to_plot):
        plt.subplot(len(sensors_to_plot), 1, idx + 1)
        # Plot up to first 500 steps for clarity
        limit = min(500, y_true.shape[0])
        plt.plot(y_true[:limit, s_idx], label='Real', alpha=0.7)
        plt.plot(y_reconstructed[:limit, s_idx], label='Reconstructed', alpha=0.7, linestyle='--')
        plt.title(f'Sensor {s_idx} - Real vs Reconstructed Data (subset)')
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "sample_sensor_reconstructions.png"))
    plt.close()

def plot_error_distribution_across_sensors(metrics, result_dir):
    plt.figure(figsize=(10, 5))
    plt.hist(metrics['per_sensor_mae'], bins=20, alpha=0.7, color='blue', label='MAE')
    plt.hist(metrics['per_sensor_rmse'], bins=20, alpha=0.7, color='orange', label='RMSE')
    plt.title('Distribution of Errors Across Sensors')
    plt.xlabel('Error (°C)')
    plt.ylabel('Frequency (Number of Sensors)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "error_distribution_across_sensors.png"))
    plt.close()

def plot_edge_vs_reconstructed_validation(y_true, y_reconstructed, edge_predictions_history, result_dir, sample_sensor=0):
    """
    Secondary validation plot comparing Edge internal prediction vs Cloud Reconstructed.
    They should diverge when an update is sent, because Cloud uses Real value, while Edge internally has mapped Real into its window but we track its raw prediction.
    """
    limit = min(500, y_true.shape[0])
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:limit, sample_sensor], label='Ground Truth', color='black', alpha=0.5)
    plt.plot(y_reconstructed[:limit, sample_sensor], label='Cloud Reconstructed', color='blue', linestyle='--', alpha=0.8)
    if edge_predictions_history is not None and len(edge_predictions_history) > 0:
        plt.plot(edge_predictions_history[:limit, sample_sensor], label='Raw Edge Prediction', color='red', linestyle=':', alpha=0.8)
        
    plt.title(f'Validation: Edge Prediction vs Cloud Reconstructed (Sensor {sample_sensor})')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "validation_edge_vs_cloud.png"))
    plt.close()
