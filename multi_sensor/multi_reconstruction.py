import numpy as np
import torch

def reconstruct_multi_sensor_cloud(
    cloud_predictions, 
    real_values, 
    transmissions
):
    """
    Cloud reconstruction logic for multiple sensors vectorized.
    The Cloud holds identical model predictions as the Edge.
    For each sensor: 
    - if data was transmitted, the Cloud uses the real_value.
    - if not, the Cloud uses its locally predicted value.
    
    Args:
        cloud_predictions (np.ndarray or list): Predicted values for N sensors by Cloud.
        real_values (np.ndarray or list): Actual values for N sensors.
        transmissions (np.ndarray or list of bool): Transmission flags for N sensors.
        
    Returns:
        np.ndarray: The reconstructed values for N sensors at current time step.
    """
    # Assuming inputs are numpy arrays or lists
    cloud_predictions = np.array(cloud_predictions)
    real_values = np.array(real_values)
    transmissions = np.array(transmissions, dtype=bool)
    
    reconstructed = np.where(transmissions, real_values, cloud_predictions)
    return reconstructed

def decide_transmissions_multi(real_values, edge_predictions, epsilon):
    """
    Vectorized transmission logic for multiple sensors.
    
    Args:
        real_values (np.ndarray): Actual values for N sensors.
        edge_predictions (np.ndarray): Edge predictions for N sensors.
        epsilon (float or None): Error threshold.
        
    Returns:
        np.ndarray (bool): Transmission flags for N sensors.
    """
    if epsilon is None:
        # None implies baseline, always transmit
        return np.ones_like(real_values, dtype=bool)
        
    errors = np.abs(real_values - edge_predictions)
    transmissions = errors > epsilon
    return transmissions
