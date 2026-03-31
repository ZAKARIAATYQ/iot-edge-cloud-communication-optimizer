import numpy as np

def generate_multi_sensor_data(base_signal, num_sensors, noise_std=0.5, random_seed=42):
    """
    Generates synthetic multi-sensor data by adding controlled Gaussian noise 
    to a base signal.
    
    Args:
        base_signal (np.ndarray): The base true signal (1D array).
        num_sensors (int): Number of sensors to simulate.
        noise_std (float): Standard deviation of the Gaussian noise.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: A 2D array of shape (time_steps, num_sensors)
    """
    np.random.seed(random_seed)
    
    # Expand base signal to multiple sensors
    time_steps = len(base_signal)
    
    # Create the base matrix (repeat base signal for each sensor)
    base_matrix = np.tile(base_signal, (num_sensors, 1)).T
    
    # Add unique, independent Gaussian noise to each sensor
    noise = np.random.normal(loc=0.0, scale=noise_std, size=(time_steps, num_sensors))
    
    multi_sensor_data = base_matrix + noise
    
    return multi_sensor_data

