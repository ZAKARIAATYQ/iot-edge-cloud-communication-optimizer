def reconstruct_cloud_data(y_real, y_pred, transmitted):
    """
    Reconstructs the data point at the cloud based on transmission decision.
    
    Args:
        y_real (float): The actual value measured by the sensor.
        y_pred (float): The predicted value at the Edge.
        transmitted (bool): Whether the Edge transmitted the real value.
        
    Returns:
        float: The value to be stored/used at the Cloud.
    """
    if transmitted:
        return y_real
    else:
        return y_pred
