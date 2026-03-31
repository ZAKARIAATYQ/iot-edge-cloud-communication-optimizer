def decide_transmission(y_real, y_pred, epsilon):
    """
    Decides whether a data point should be transmitted to the cloud based on the prediction error.
    
    Args:
        y_real (float): The actual measured value at the Edge.
        y_pred (float): The predicted value by the Edge model.
        epsilon (float): The threshold for the prediction error.
        
    Returns:
        bool: True if data should be transmitted, False otherwise.
    """
    error = abs(y_real - y_pred)
    if epsilon is None:
        # None implies baseline, always transmit
        return True
    
    return error > epsilon
