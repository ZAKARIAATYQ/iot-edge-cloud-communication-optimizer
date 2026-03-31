import os
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    GRU Model definition identical to the original trained model.
    - input_size: 1
    - hidden_size: 64
    - num_layers: 1
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x expected shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Taking the last time step output to pass into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

def main():
    # File paths
    original_model_path = os.path.join('model', 'window72.pth')
    quantized_model_path = os.path.join('model', 'window72_tiny.pth')
    
    print(f"Loading original model from: {original_model_path}...")
    
    # 1. Initialize the same architecture
    model = GRUModel(input_size=1, hidden_size=64, num_layers=1)
    
    # 2. Load the trained state dictionary on CPU
    try:
        # We try weights_only first for security
        state_dict = torch.load(original_model_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception:
        # Fallback if the previous save was a full model or needs older loading compatibilities
        state_dict = torch.load(original_model_path, map_location=torch.device('cpu'))

    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()
        
    model.load_state_dict(state_dict)
    print("Original model loaded successfully.")

    # 3. Set model to evaluation mode (required before dynamic quantization)
    model.eval()
    print("Model set to evaluation mode (.eval()).")

    print("Applying PyTorch dynamic quantization...")
    print("Targeting: torch.nn.GRU, torch.nn.Linear (dtype=torch.qint8)...")
    
    # 4. Apply Dynamic Quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.GRU, nn.Linear}, 
        dtype=torch.qint8
    )
    
    print("Quantization complete!")

    # 5. Save the quantized model's state dictionary
    print(f"Saving quantized model to: {quantized_model_path}...")
    torch.save(quantized_model.state_dict(), quantized_model_path)
    
    print("Quantized model saved successfully.")
    
    # Bonus check to show sizes
    if os.path.exists(original_model_path) and os.path.exists(quantized_model_path):
        orig_size = os.path.getsize(original_model_path) / 1024
        quant_size = os.path.getsize(quantized_model_path) / 1024
        print(f"\n--- Size Comparison ---")
        print(f"Original Model:  {orig_size:.2f} KB")
        print(f"Quantized Model: {quant_size:.2f} KB")
        print(f"Reduction:       {(1 - quant_size/orig_size)*100:.1f}%")

if __name__ == '__main__':
    main()
