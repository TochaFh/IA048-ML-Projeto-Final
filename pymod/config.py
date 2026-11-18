from pathlib import Path

RANDOM_SEED = 13

# paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"


import torch

DEVICE = "cpu"

# Define device based on your hardware availability
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # For Apple Silicon Macbooks (M1, M2, M3, M4 etc.)

print(f"Using device: {DEVICE}")