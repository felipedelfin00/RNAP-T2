import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT
)

from .download_dataset import download_dataset
from .utils import seed_worker

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_dataset():
    dataset_root = download_dataset()

    return datasets.ImageFolder(
        root = dataset_root,
        transform = BASE_TRANSFORM
    )


def split_dataset(dataset, seed):
    idx = list(range(len(dataset)))
    labels = dataset.targets

    train_idx, temp_idx = train_test_split(
        idx,
        train_size = TRAIN_SPLIT,
        stratify = labels,
        random_state = seed
    )

    relative_test_size = TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size = relative_test_size,
        stratify = [labels[i] for i in temp_idx],
        random_state = seed
    )

    return train_idx, val_idx, test_idx


def calculate_normalization_stats(dataset, train_idx):
    dataset.transform = BASE_TRANSFORM

    loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY
    )

    channel_sum = torch.zeros(3, dtype = torch.float64)
    channel_squared_sum = torch.zeros(3, dtype = torch.float64)
    total_pixels = 0

    for images, _ in loader:
        images = images.to(torch.float64)

        channel_sum += images.sum(dim = (0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim = (0, 2, 3))

        total_pixels += (images.size(0) * images.size(2) * images.size(3))

    mean = channel_sum / total_pixels

    variance = (channel_squared_sum / total_pixels) - (mean ** 2)

    std = torch.sqrt(variance.clamp(min = 0))

    if torch.any(std == 0):
        raise ValueError("A channel has zero standard deviation.")

    return mean.tolist(), std.tolist()


def apply_normalization(dataset, mean, std):
    dataset.transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])


def create_dataloaders(dataset, train_idx, val_idx, test_idx, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    common_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": NUM_WORKERS > 0,
        "worker_init_fn": seed_worker
    }

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        shuffle = True,
        generator = generator,
        **common_args
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        shuffle = False,
        **common_args
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        shuffle = False,
        **common_args
    )

    return train_loader, val_loader, test_loader


def select_train_fraction(train_idx, labels, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be between 0 and 1.")

    if fraction == 1.0:
        return train_idx

    selected_idx, _ = train_test_split(
        train_idx,
        train_size = fraction,
        stratify = [labels[i] for i in train_idx],
        random_state = seed
    )

    return selected_idx