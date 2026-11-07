import json
import pandas as pd
import time
import torch

from src.config import (
    DATA_FRACTIONS,
    EPOCHS,
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
    apply_normalization,
    create_dataloaders,
    select_train_fraction
)

from src.model import ArchitectureCNN

from src.train import (
    train_model,
    benchmark_training
)

from src.evaluate import evaluate_model
from src.utils import set_seed


def prepare_seed_data(dataset, seed):
    train_idx, val_idx, test_idx = split_dataset(
        dataset,
        seed
    )

    mean, std = calculate_normalization_stats(
        dataset,
        train_idx
    )

    apply_normalization(
        dataset,
        mean,
        std
    )

    return train_idx, val_idx, test_idx


def create_loaders(
    dataset,
    train_idx,
    val_idx,
    test_idx,
    fraction,
    seed
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
        seed
    )


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

    train_idx, val_idx, test_idx = first_seed_data

    benchmark_times = {}

    for fraction in DATA_FRACTIONS:
        fraction_label = int(fraction * 100)

        print(f"\nMeasuring training with {fraction_label}% of the data")

        train_loader, val_loader, _ = create_loaders(
            dataset,
            train_idx,
            val_idx,
            test_idx,
            fraction,
            seed
        )

        set_seed(seed)

        average_epoch_time = benchmark_training(
            lambda: ArchitectureCNN(NUM_CLASSES),
            train_loader,
            val_loader,
            device = DEVICE
        )

        benchmark_times[fraction] = average_epoch_time

        print (
            f"Average time per epoch: "
            f"{format_duration(average_epoch_time)}"
        )

    print("\nESTIMATED TOTAL TIME")

    for epochs in [15, 25, EPOCHS]:
        total_seconds = (
            len(SEEDS) * epochs * sum(benchmark_times.values())
        )

        print(
            f"{epochs} epochs per execution:"
            f"{format_duration(total_seconds)}"
        )


def run_experiments(dataset, class_names, first_seed_data):
    results = []

    for seed in SEEDS:
        print ("\n" + "-" * 48)
        print(f"SEED {seed}")
        print("-" * 48)

        if seed == SEEDS[0]:
            train_idx, val_idx, test_idx = first_seed_data
        else:
            train_idx, val_idx, test_idx = prepare_seed_data(
                dataset,
                seed
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
                fraction,
                seed
            )

            set_seed(seed)

            model = ArchitectureCNN(NUM_CLASSES).to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)

            start_time = time.perf_counter()

            history, best_epoch = train_model(
                model,
                train_loader,
                val_loader,
                device = DEVICE,
                checkpoint_path = (CHECKPOINTS_DIR / f"seed_{seed}_fraction_{fraction:.2f}.pth")
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

            run_name = f"seed_{seed}_fraction_{fraction_label}"

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
                "seed": seed,
                "fraction": fraction,
                "best_epoch": best_epoch,
                "epochs_trained": len(history),
                "test_loss": evaluation["loss"],
                "test_accuracy": evaluation["accuracy"],
                "test_f1_macro": evaluation["f1_macro"],
                "training_time_seconds": training_time
            }

            results.append(result)

            print("Configuration finished!")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "summary.csv",
        index = False
    )

    print ("\n" + "-" * 48)
    print("RESUME")
    print("-" * 48)

    print(results_df.to_string(index = False))

    total_training_time = results_df["training_time_seconds"].sum()
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