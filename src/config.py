import torch

from pathlib import Path


#   Directories
ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"


#   Dataset
IMAGE_SIZE = 128
NUM_CLASSES = 25

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15


#   Training
BATCH_SIZE = 64
DATA_FRACTIONS = [0.25, 1.0]
EPOCHS = 35
LEARNING_RATE = 0.001
PATIENCE = 5
SEEDS = [6, 12, 18, 24, 30]
WEIGHT_DECAY = 0.0001
USE_AMP = True
DETERMINISTIC = False

RESNET_LEARNING_RATE = 0.0001
RESNET_WEIGHT_DECAY = 0.001
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]


#   Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)