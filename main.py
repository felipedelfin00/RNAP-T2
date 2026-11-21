import json
import pandas as pd
import time
import torch

from src.config import (
    DATA_FRACTIONS,
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    RESNET_LEARNING_RATE,
    RESNET_WEIGHT_DECAY,
    RESNET_MEAN,
    RESNET_STD,
    NUM_CLASSES,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    SEEDS,
    DEVICE
)

from src.dataset import (
    load_dataset,
    split_dataset,
    calculate_normalization_stats,
    create_dataloaders,
    select_train_fraction
)

from src.model import (
    ArchitectureCNN,
    ArchitectureResNet18
)

from src.train import (
    train_model,
    benchmark_training
)

from src.evaluate import evaluate_model
from src.utils import set_seed


MODEL_CONFIGS = [
    {
        "name": "baseline",
        "model_class": ArchitectureCNN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "normalization": "dataset"
    },
    {
        "name": "resnet18",
        "model_class": ArchitectureResNet18,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": RESNET_LEARNING_RATE,
        "patience": PATIENCE,
        "weight_decay": RESNET_WEIGHT_DECAY,
        "normalization": "imagenet"
    }
]


def prepare_seed_data(dataset, seed):
    train_idx, val_idx, test_idx = split_dataset(
        dataset,
        seed
    )

    mean, std = calculate_normalization_stats(
        dataset,
        train_idx
    )

    return train_idx, val_idx, test_idx, mean, std


def create_loaders(
    dataset,
    train_idx,
    val_idx,
    test_idx,
    mean,
    std,
    fraction,
    seed,
    batch_size
):
    selected_train_idx = select_train_fraction(
        train_idx,
        dataset.targets,
        fraction,
        seed
    )

    return create_dataloaders(
        dataset,
        selected_train_idx,
        val_idx,
        test_idx,
        mean,
        std,
        seed,
        batch_size
    )


def get_normalization(model_config, seed_data):
    if model_config["normalization"] == "imagenet":
        return RESNET_MEAN, RESNET_STD

    _, _, _, mean, std = seed_data

    return mean, std


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}min {remaining_seconds}s"

    if minutes > 0:
        return f"{minutes}min {remaining_seconds}s"

    return f"{remaining_seconds}s"


def run_benchmark(dataset, seed, first_seed_data):
    print("\n" + "-" * 48)
    print("BENCHMARK")
    print("-" * 48)

    train_idx, val_idx, test_idx, _, _ = first_seed_data

    benchmark_times = {}

    for model_config in MODEL_CONFIGS:
        model_name = model_config["name"]

        print("\n" + "=" * 48)
        print(f"MODEL: {model_name}")
        print("=" * 48)

        benchmark_times[model_name] = {}

        mean, std = get_normalization(
            model_config,
            first_seed_data
        )

        for fraction in DATA_FRACTIONS:
            fraction_label = int(fraction * 100)

            print(f"\nMeasuring training with {fraction_label}% of the data")

            train_loader, val_loader, _ = create_loaders(
                dataset,
                train_idx,
                val_idx,
                test_idx,
                mean,
                std,
                fraction,
                seed,
                model_config["batch_size"]
            )

            set_seed(seed)

            average_epoch_time = benchmark_training(
                lambda: model_config["model_class"](NUM_CLASSES),
                train_loader,
                val_loader,
                device = DEVICE,
                learning_rate=model_config["learning_rate"],
                weight_decay=model_config["weight_decay"]
            )

            benchmark_times[model_name][fraction] = average_epoch_time

            print (
                f"Average time per epoch: "
                f"{format_duration(average_epoch_time)}"
            )

            del train_loader, val_loader

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    print("\nESTIMATED TOTAL TIME")

    for model_config in MODEL_CONFIGS:
        model_name = model_config["name"]

        for epochs in [15, 25, model_config["epochs"]]:
            total_seconds = (
                len(SEEDS) 
                * epochs 
                * sum(benchmark_times[model_name].values())
            )

            print(
                f"{model_name} - {epochs} epochs per execution:"
                f"{format_duration(total_seconds)}"
            )


def run_experiments(dataset, class_names, first_seed_data):
    all_results = []

    for model_config in MODEL_CONFIGS:
        model_name = model_config["name"]

        model_results = []

        print("\n" + "-" * 48)
        print(f"MODEL: {model_name}")
        print("-" * 48)

        for seed in SEEDS:
            print ("\n" + "-" * 48)
            print(f"SEED {seed}")
            print("-" * 48)

            if seed == SEEDS[0]:
                seed_data = first_seed_data

            else:
                seed_data = prepare_seed_data(
                    dataset,
                    seed
                )

            train_idx, val_idx, test_idx, _, _ = seed_data

            mean, std = get_normalization(
                model_config,
                seed_data
            )

            for fraction in DATA_FRACTIONS:
                fraction_label = int(fraction * 100)

                print ("\n" + "-" * 48)
                print(f"{fraction_label}% of the data")
                print("-" * 48)

                train_loader, val_loader, test_loader = create_loaders(
                    dataset,
                    train_idx,
                    val_idx,
                    test_idx,
                    mean,
                    std,
                    fraction,
                    seed,
                    model_config["batch_size"]
                )

                set_seed(seed)

                model = model_config["model_class"](
                    NUM_CLASSES
                ).to(
                    DEVICE,
                    memory_format=torch.contiguous_format
                )

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize(DEVICE)

                start_time = time.perf_counter()

                run_name = (
                    f"{model_name}_seed_{seed}_"
                    f"fraction_{fraction:.2f}"
                )

                checkpoint_path = (
                    CHECKPOINTS_DIR / f"{run_name}.pth"
                )

                history, best_epoch = train_model(
                    model,
                    train_loader,
                    val_loader,
                    device=DEVICE,
                    epochs=model_config["epochs"],
                    learning_rate=model_config["learning_rate"],
                    patience=model_config["patience"],
                    weight_decay=model_config["weight_decay"],
                    checkpoint_path=checkpoint_path 
                )

                print("Training finished successfully.")

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize(DEVICE)

                training_time = time.perf_counter() - start_time

                evaluation = evaluate_model(
                    model,
                    test_loader,
                    class_names,
                    device = DEVICE
                )

                print("Test evaluation finished successfully.")

                pd.DataFrame(history).to_csv(
                    RESULTS_DIR / f"{run_name}_history.csv",
                    index = False
                )

                pd.DataFrame(
                    evaluation["confusion_matrix"],
                    index = class_names,
                    columns = class_names
                ).to_csv(
                    RESULTS_DIR / f"{run_name}_confusion_matrix.csv"
                )

                with open(
                    RESULTS_DIR / f"{run_name}_classification_report.json",
                    "w",
                    encoding = "utf-8"
                ) as file :
                    json.dump(
                        evaluation["classification_report"],
                        file,
                        indent = 4,
                        ensure_ascii = False
                    )

                result = {
                    "model": model_name,
                    "seed": seed,
                    "fraction": fraction,
                    "best_epoch": best_epoch,
                    "epochs_trained": len(history),
                    "test_loss": evaluation["loss"],
                    "test_accuracy": evaluation["accuracy"],
                    "test_f1_macro": evaluation["f1_macro"],
                    "training_time_seconds": training_time
                }

                model_results.append(result)
                all_results.append(result)

                print("Configuration finished!")

                del model
                del train_loader
                del val_loader
                del test_loader

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        results_df = pd.DataFrame(model_results)

        results_df.to_csv(
            RESULTS_DIR / f"summary_{model_name}.csv",
            index = False
        )

        print ("\n" + "-" * 48)
        print(f"SUMMARY: {model_name}")
        print("-" * 48)

        print(results_df.to_string(index = False))

        total_training_time = results_df["training_time_seconds"].sum()

        print(
            f"\nTotal training time: "
            f"{format_duration(total_training_time)}"
        )

    all_results_df = pd.DataFrame(all_results)

    all_results_df.to_csv(
        RESULTS_DIR / "summary_all.csv",
        index=False
    )

    print("\n" + "-" * 48)
    print("FINAL SUMMARY")
    print("-" * 48)

    print(all_results_df.to_string(index=False))

    total_training_time = (
        all_results_df["training_time_seconds"].sum()
    )

    print(
        f"\nTotal training time: "
        f"{format_duration(total_training_time)}"
    )


def main():
    RESULTS_DIR.mkdir(
        parents = True,
        exist_ok = True
    )

    CHECKPOINTS_DIR.mkdir(
        parents = True,
        exist_ok = True
    )

    print("------ Architectural Style Classification ------")

    dataset = load_dataset()
    class_names = dataset.classes

    print(f"Images found: {len(dataset)}")
    print(f"Classes found: {len(class_names)}")

    print ("\n" + "-" * 48)

    first_seed_data = prepare_seed_data(dataset, SEEDS[0])

    run_benchmark_choice = input(
        "Run benchmark before training? (y/n): "
    ).strip().lower()

    if run_benchmark_choice in ("y", "s", "yes", "sim"):
        run_benchmark(dataset, SEEDS[0], first_seed_data)
        input("\nPress ENTER to start the experiments...")

    elif run_benchmark_choice not in ("n", "no", "nao", "não"):
        print("Invalid option. Skipping benchmark.")

    print ("\n" + "-" * 48)

    run_experiments(
        dataset,
        class_names,
        first_seed_data
    )


if __name__ == "__main__":
    main()