import torch

from pathlib import Path


#   Directories
ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "data"

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"


#   Dataset
IMAGE_SIZE = 224
NUM_CLASSES = 25

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

NUM_WORKERS = 2


#   Training
BATCH_SIZE = 32
DATA_FRACTIONS = [0.25, 1.0]
EPOCHS = 35
LEARNING_RATE = 0.001
PATIENCE = 7
SEEDS = [6, 12, 18, 24, 30]
WEIGHT_DECAY = 0.0001


#   Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PIN_MEMORY = DEVICE.type == "cuda"
