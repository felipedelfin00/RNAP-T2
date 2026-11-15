import hashlib
import os
import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import datasets

from .config import (
    CACHE_DIR,
    DEVICE,
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT
)

from .download_dataset import download_dataset


class CachedDataset:
    def __init__(self, images, targets, classes):
        self.images = images
        self.targets = list(targets)
        self.classes = list(classes)
        self.labels = torch.as_tensor(self.targets, dtype = torch.long)

    def __len__(self):
        return len(self.targets)

    def to(self, device):
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)

        return self


class TensorLoader:
    def __init__(
        self,
        dataset,
        indices,
        mean,
        std,
        batch_size = BATCH_SIZE,
        shuffle = False,
        generator = None
    ):
        device = dataset.images.device

        self.images = dataset.images
        self.labels = dataset.labels
        self.indices = torch.as_tensor(list(indices), dtype = torch.long)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.num_samples = len(self.indices)

        self.mean = torch.tensor(
            mean, dtype = torch.float32, device = device
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            std, dtype = torch.float32, device = device
        ).view(1, 3, 1, 1)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1)

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(
                self.num_samples,
                generator = self.generator
            )
            order = self.indices[permutation]
        else:
            order = self.indices

        order = order.to(self.images.device)

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = order[start:start + self.batch_size]

            images = self.images[batch_idx].float().div_(255.0)
            images = (images - self.mean) / self.std
            images = images.contiguous(memory_format = torch.contiguous_format)

            yield images, self.labels[batch_idx]


def _build_signature(samples):
    digest = hashlib.sha1()
    digest.update(f"size={IMAGE_SIZE}".encode())

    for path, label in samples:
        digest.update(
            f"{os.path.basename(os.path.dirname(path))}/"
            f"{os.path.basename(path)}|{os.path.getsize(path)}|{label}"
            .encode()
        ) 

    return digest.hexdigest()


def _load_resized(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        return np.array(image, dtype = np.uint8)


def _build_image_tensor(paths):
    images = torch.empty(
        (len(paths), 3, IMAGE_SIZE, IMAGE_SIZE),
        dtype = torch.uint8
    )

    workers = max(1, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers = workers) as executor:
        for i, array in enumerate(executor.map(_load_resized, paths)):
            images[i] = torch.from_numpy(array).permute(2, 0, 1)

            if (i + 1) % 1000 == 0 or (i + 1) == len(paths):
                print(f"Cached {i + 1}/{len(paths)} images")

    return images
 

def load_dataset(device = DEVICE):
    dataset_root = download_dataset()

    folder = datasets.ImageFolder(root = dataset_root)

    signature = _build_signature(folder.samples)
    cache_path = CACHE_DIR / f"images_{IMAGE_SIZE}.pt"

    if cache_path.exists():
        cached = torch.load(cache_path, map_location = "cpu")

        if cached["signature"] == signature:
            print(f"Image cache loaded: {cache_path}")

            return CachedDataset(
                cached["images"],
                cached["targets"],
                cached["classes"]
            ).to(device)

        print("Image cache is outdated. Rebuilding.")

    print(f"Building image cache ({IMAGE_SIZE}x{IMAGE_SIZE}).")

    images = _build_image_tensor([path for path, _ in folder.samples])

    CACHE_DIR.mkdir(parents = True, exist_ok = True)

    temporary_path = cache_path.with_suffix(".tmp")

    torch.save({
        "signature": signature,
        "images": images,
        "targets": list(folder.targets),
        "classes": list(folder.classes)
    }, temporary_path)

    os.replace(temporary_path, cache_path)

    return CachedDataset(
        images,
        folder.targets,
        folder.classes
    ).to(device)


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


def calculate_normalization_stats(dataset, train_idx, chunk_size = 512):
    device = dataset.images.device
    indices = torch.as_tensor(list(train_idx), dtype = torch.long)

    channel_sum = torch.zeros(3, dtype = torch.float64, device = device)
    channel_squared_sum = torch.zeros(3, dtype = torch.float64, device = device)
    total_pixels = 0

    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start:start + chunk_size].to(device)
        
        images = dataset.images[chunk_idx].to(torch.float64) / 255.0

        channel_sum += images.sum(dim = (0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim = (0, 2, 3))

        total_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / total_pixels

    variance = (channel_squared_sum / total_pixels) - (mean ** 2)

    std = torch.sqrt(variance.clamp(min = 0))

    if torch.any(std == 0):
        raise ValueError("A channel has zero standard deviation.")

    return mean.cpu().tolist(), std.cpu().tolist()


def create_dataloaders(dataset, train_idx, val_idx, test_idx, mean, std, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)


    train_loader = TensorLoader(
        dataset,
        train_idx,
        mean,
        std,
        shuffle = True,
        generator = generator
    )

    val_loader = TensorLoader(dataset, val_idx, mean, std)
    test_loader = TensorLoader(dataset, test_idx, mean, std)

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