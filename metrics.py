import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_reconstructed, total_points, transmitted_points):
    """
    Computes communication and error metrics.
    """
    reduction = (1 - transmitted_points / total_points) * 100 if total_points > 0 else 0
    
    mae = mean_absolute_error(y_true, y_reconstructed)
    rmse = np.sqrt(mean_squared_error(y_true, y_reconstructed))
    r2 = r2_score(y_true, y_reconstructed)
    
    return {
        "total": total_points,
        "transmitted": transmitted_points,
        "reduction": reduction,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def analyze_critical_events(y_true, y_reconstructed, transmission_flags, threshold=2.0):
    """
    Detect large temperature changes (|ΔT| > 2°C) and check if missed by filtering.
    """
    critical_events = 0
    missed_events = 0
    
    for i in range(1, len(y_true)):
        delta_t = abs(y_true[i] - y_true[i-1])
        if delta_t > threshold:
            critical_events += 1
            # Check if this critical point was NOT transmitted
            if not transmission_flags[i]:
                # It means it was reconstructed using prediction and was not explicitly sent
                missed_events += 1
                
    return critical_events, missed_events

def save_metrics(result_dir, metrics, critical_events, missed_events):
    filepath = os.path.join(result_dir, "metrics.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("--- SIMULATION METRICS ---\n")
        f.write(f"Total Points: {metrics['total']}\n")
        f.write(f"Transmitted Points: {metrics['transmitted']}\n")
        f.write(f"Transmission Reduction: {metrics['reduction']:.2f}%\n")
        f.write("\n")
        f.write("--- RECONSTRUCTION QUALITY ---\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"R2: {metrics['r2']:.4f}\n")
        f.write("\n")
        f.write("--- CRITICAL EVENTS (|ΔT| > 2°C) ---\n")
        f.write(f"Total Critical Events: {critical_events}\n")
        f.write(f"Missed Events (Not Transmitted): {missed_events}\n")

def plot_real_vs_reconstructed(y_true, y_reconstructed, result_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:500], label='Real', alpha=0.7)  # Plotting subset for visibility
    plt.plot(y_reconstructed[:500], label='Reconstructed', alpha=0.7, linestyle='--')
    plt.title('Real vs Reconstructed Data (subset)')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "real_vs_reconstructed.png"))
    plt.close()

def plot_error_boxplot(y_true, y_reconstructed, result_dir):
    errors = np.abs(np.array(y_true) - np.array(y_reconstructed))
    plt.figure(figsize=(6, 8))
    plt.boxplot(errors)
    plt.title('Reconstruction Error Boxplot')
    plt.ylabel('Absolute Error (°C)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "error_boxplot.png"))
    plt.close()

def plot_residuals(y_true, y_reconstructed, result_dir):
    residuals = np.array(y_true) - np.array(y_reconstructed)
    plt.figure(figsize=(12, 6))
    plt.plot(residuals[:500], label='Residuals', color='red', alpha=0.6)
    plt.title('Residuals (Real - Reconstructed) (subset)')
    plt.xlabel('Time Step')
    plt.ylabel('Error (°C)')
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "residuals.png"))
    plt.close()
