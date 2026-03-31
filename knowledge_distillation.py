"""
knowledge_distillation.py
--------------------------
Standalone Knowledge Distillation training script.

Trains a student GRU (TinyML) using soft distillation from the full
teacher GRU, combining ground-truth loss with teacher-imitation loss.

Usage:
    python knowledge_distillation.py

Output:
    model/window72_tiny_distilled.pth
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SENSOR_DIR = os.path.join(PROJECT_ROOT, "5_SENSOR")
TEACHER_PATH = os.path.join(PROJECT_ROOT, "model", "window72.pth")
STUDENT_PATH = os.path.join(PROJECT_ROOT, "model", "window72_tiny.pth")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "model", "window72_tiny_distilled.pth")

WINDOW_SIZE = 72
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
ALPHA = 0.7          # weight for ground-truth loss (1 - ALPHA for teacher loss)
VAL_SPLIT = 0.1      # fraction of data used for validation

SENSOR_FILES = [
    ("AZILALSENSOR.csv",        "Azilal"),
    ("BENIMELLALSENSOR.csv",    "Beni Mellal"),
    ("FQUIHBENSALAHSENSOR.csv", "Fquih Ben Salah"),
    ("KHENIFRASENSOR.csv",      "Khenifra"),
    ("MARAKKECHSENSOR.csv",     "Marrakech"),
]


# ---------------------------------------------------------------------------
# GRU Model (same architecture as the rest of the project)
# ---------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------
class SlidingWindowDataset(Dataset):
    """Create (window, target) pairs from a 1-D scaled temperature array."""

    def __init__(self, scaled_data, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.data = scaled_data
        self.length = len(scaled_data) - window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]                # (W, 1)
        y = self.data[idx + self.window_size]                      # (1,)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Data loading (mirrors existing pipeline)
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    """
    Load all 5 sensor CSVs, convert K to C, concatenate into one long series,
    apply MinMaxScaler, and build sliding-window train/val datasets.
    """
    all_celsius = []

    for filename, city in SENSOR_FILES:
        filepath = os.path.join(SENSOR_DIR, filename)
        df = pd.read_csv(filepath, parse_dates=["valid_time"])
        temps = df["t2m"].values.astype(np.float64)
        if temps.mean() > 200:
            temps = temps - 273.15
        all_celsius.append(temps)
        print(f"  Loaded {city}: {len(temps)} samples "
              f"[{temps.min():.1f} C .. {temps.max():.1f} C]")

    # Concatenate all sensors into a single long series
    combined = np.concatenate(all_celsius)
    print(f"\n  Combined dataset: {len(combined)} samples")

    # Global MinMaxScaler (same approach as simulation)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(combined.reshape(-1, 1))  # (N, 1)

    # Train / Val split
    n = len(scaled) - WINDOW_SIZE
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size

    # Split at boundary keeping window continuity
    train_data = scaled[: train_size + WINDOW_SIZE]
    val_data = scaled[train_size : ]

    train_ds = SlidingWindowDataset(train_data, WINDOW_SIZE)
    val_ds = SlidingWindowDataset(val_data, WINDOW_SIZE)

    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    return train_ds, val_ds, scaler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_distillation(teacher, student, train_loader, val_loader, epochs=EPOCHS):
    """
    Knowledge distillation: student learns from both ground truth and teacher.

    loss = alpha * MSE(student, real) + (1 - alpha) * MSE(student, teacher)
    """
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    mse = nn.MSELoss()

    teacher.eval()

    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            # Teacher prediction (frozen)
            with torch.no_grad():
                teacher_pred = teacher(x_batch)      # (B, 1)

            # Student prediction
            student_pred = student(x_batch)           # (B, 1)

            # Combined distillation loss
            loss_real = mse(student_pred, y_batch)
            loss_teacher = mse(student_pred, teacher_pred)
            loss = ALPHA * loss_real + (1.0 - ALPHA) * loss_teacher

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        student.eval()
        val_mae_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                pred_val = student(x_val)
                val_mae_sum += torch.sum(torch.abs(pred_val - y_val)).item()
                val_count += y_val.numel()

        val_mae = val_mae_sum / max(val_count, 1)

        print(f"  Epoch {epoch:>2}/{epochs}  |  "
              f"Train Loss: {avg_train_loss:.6f}  |  "
              f"Val MAE (scaled): {val_mae:.6f}")

    return student


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  KNOWLEDGE DISTILLATION - Teacher -> Student (Tiny GRU)")
    print("=" * 65)

    # 1. Load data
    print("\n[1/4] Loading sensor data...")
    train_ds, val_ds, scaler = load_and_prepare_data()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load teacher (full float32 GRU — no quantization)
    print("\n[2/4] Loading TEACHER model...")
    teacher = GRUModel(input_size=1, hidden_size=64, num_layers=1)
    teacher.load_state_dict(
        torch.load(TEACHER_PATH, map_location="cpu")
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("  Teacher loaded and frozen.")

    # 3. Load student (fresh float GRU - NOT quantized during training)
    #    We initialise from a fresh model so gradients flow normally;
    #    the student will be quantized AFTER distillation if desired.
    print("\n[3/4] Initialising STUDENT model (float32 for training)...")
    student = GRUModel(input_size=1, hidden_size=64, num_layers=1)

    # Try to warm-start from the existing tiny model float weights
    try:
        tmp = GRUModel(input_size=1, hidden_size=64, num_layers=1)
        tmp = torch.quantization.quantize_dynamic(
            tmp, {nn.GRU, nn.Linear}, dtype=torch.qint8
        )
        tmp.load_state_dict(
            torch.load(STUDENT_PATH, map_location="cpu", weights_only=False)
        )
        student_sd = student.state_dict()
        loaded = 0
        for name, param in tmp.named_parameters():
            if name in student_sd and student_sd[name].shape == param.shape:
                student_sd[name] = param.data.clone()
                loaded += 1
        student.load_state_dict(student_sd)
        print(f"  Warm-started {loaded} parameter(s) from existing tiny model.")
    except Exception as e:
        print(f"  Could not warm-start from tiny model ({e}). Using fresh init.")

    student.train()
    print(f"  Student ready (float32, trainable).\n")

    # 4. Distillation training
    print("[4/4] Running Knowledge Distillation...\n")
    student = train_distillation(teacher, student, train_loader, val_loader)

    # Save distilled student (float state_dict - can be quantized later)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(student.state_dict(), OUTPUT_PATH)
    print(f"\n  Distilled student saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 65)
    print("  KNOWLEDGE DISTILLATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
