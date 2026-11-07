import kagglehub

from pathlib import Path

from .config import (
    DATASET_DIR,
    NUM_CLASSES
)


DATASET_HANDLE = "dumitrux/architectural-styles-dataset"
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}


def find_dataset_root(search_dir):
    search_dir = Path(search_dir)

    candidates = [
        search_dir,
        *[path for path in search_dir.rglob("*") if path.is_dir()]
    ]

    for candidate in candidates:
        class_dirs = [
            path for path in candidate.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]

        valid_classes = [
            directory
            for directory in class_dirs
            if any(
                file.suffix.lower() in IMAGE_EXTENSIONS
                for file in directory.rglob("*")
                if file.is_file()
            )
        ]

        if len(valid_classes) == NUM_CLASSES:
            print(f"Dataset root found: {candidate}")
            return candidate

    raise FileNotFoundError(f"Could not find a directory containing {NUM_CLASSES} classes.")

def download_dataset():
    DATASET_DIR.mkdir(parents = True, exist_ok = True)

    try:
        return find_dataset_root(DATASET_DIR)
    except FileNotFoundError:
        pass

    print("Downloading dataset.")

    downloaded = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir = str(DATASET_DIR)
    )

    print("Download finished.")
    return find_dataset_root(downloaded)

if __name__ == "__main__":
    download_dataset()