from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "data"
WINDOW_SIZE = 5
BATCH_SIZE = 42
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LR = 0.001
EPOCHS = 50
MODEL_PATH = BASE_DIR / "models" / "model.pth"
