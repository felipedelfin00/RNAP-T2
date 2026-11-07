import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.ticker import MaxNLocator

from src.config import RESULTS_DIR, PLOTS_DIR

TRAIN_COLOR = "#1F77B4"
VAL_COLOR = "#D62728"
RUNS_COLOR = "#9E9E9E"

sns.set_theme(style = "whitegrid", context = "notebook")


def plot_confusion_matrix(csv_path):
    matrix_df = pd.read_csv(csv_path, index_col = 0)

    counts = matrix_df.values
    n_classes = len(matrix_df)

    row_sums = counts.sum(axis = 1, keepdims = True)
    normalized = np.divide(
        counts,
        row_sums,
        out = np.zeros_like(counts, dtype = float),
        where = row_sums > 0
    )
    annotations = np.where(counts > 0, counts.astype(str), "")

    cell = 0.6
    side = max(8, n_classes * cell)
    annot_fontsize = max(6, min(12, 220 / max(n_classes, 1)))
    tick_fontsize = max(7, min(12, 260 / max(n_classes, 1)))

    fig, ax = plt.subplots(figsize = (side + 3, side + 1))

    sns.heatmap(
        normalized,
        ax = ax, 
        annot = annotations,
        fmt = "",
        annot_kws = {"fontsize": annot_fontsize},
        cmap = "Blues",
        vmin = 0,
        vmax = 1,
        square = True,
        linewidths = 0.5,
        linecolor = "whitesmoke",
        xticklabels = matrix_df.columns,
        yticklabels = matrix_df.index,
        cbar_kws = {"label": "Fraction of true class", "shrink": 0.7}
    )

    ax.set_xlabel("Predicted label", fontsize=13, labelpad=10)
    ax.set_ylabel("True label", fontsize=13, labelpad=10)
    ax.set_title(
        csv_path.stem.replace("_", " ").title(),
        fontsize=16,
        pad=16
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelrotation=0, labelsize=tick_fontsize)

    output_path = PLOTS_DIR / f"{csv_path.stem}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved: {output_path}")


def load_histories(fraction_label):
    history_files = sorted(
        RESULTS_DIR.glob(f"seed_*_fraction_{fraction_label}_history.csv")
    )

    if not history_files:
        raise FileNotFoundError(
            f"No history files found for fraction {fraction_label}%."
        )

    histories = [pd.read_csv(csv_path) for csv_path in history_files]

    return pd.concat(histories, ignore_index=True)


def calculate_summary(summary_df, fraction):
    fraction_df = summary_df[summary_df["fraction"].round(2) == fraction]
    
    if fraction_df.empty:
        raise ValueError(f"No summary results for fraction {fraction:.0%}.")

    metrics = {}

    for column in ["test_f1_macro", "test_accuracy", "training_time_seconds"]:
        metrics[column] = {
            "mean": fraction_df[column].mean(),
            "std": fraction_df[column].std()
        }

    return metrics


def format_mean_std(mean, std, decimals = 4):
    if pd.isna(std):
        return f"{mean:.{decimals}f} ± N/A"

    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_metrics_text(metrics):
    f1 = metrics["test_f1_macro"]
    acc = metrics["test_accuracy"]
    time_s = metrics["training_time_seconds"]

    return (
        f"Test F1 macro: {format_mean_std(f1['mean'], f1['std'])}   |   "
        f"Accuracy: {format_mean_std(acc['mean'], acc['std'])}   |   "
        f"Time: {format_mean_std(time_s['mean'], time_s['std'], 1)} s"
    )

def format_n_runs(counts):
    first, last = int(counts.iloc[0]), int(counts.iloc[-1])

    if first == last:
        return f"n = {first} runs"

    return f"n = {first} → {last} runs"


def draw_loss_panel(ax, history_df):
    grouped = history_df.groupby("epoch")
    
    mean_train = grouped["train_loss"].mean()
    std_train = grouped["train_loss"].std()
    mean_val = grouped["val_loss"].mean()
    std_val = grouped["val_loss"].std()
    counts = grouped["train_loss"].count()

    ax.plot(
        mean_train.index, mean_train,
        color=TRAIN_COLOR, label="Training", linewidth=2.2
    )
    ax.plot(
        mean_val.index, mean_val,
        color=VAL_COLOR, label="Validation", linewidth=2.2
    )

    if std_train.notna().any():
        ax.fill_between(
            mean_train.index,
            (mean_train - std_train).clip(lower=0),
            mean_train + std_train,
            color=TRAIN_COLOR, alpha=0.18, linewidth=0
        )

    if std_val.notna().any():
        ax.fill_between(
            mean_val.index,
            (mean_val - std_val).clip(lower=0),
            mean_val + std_val,
            color=VAL_COLOR, alpha=0.18, linewidth=0
        )

    best_epoch = mean_val.idxmin()
    best_value = mean_val.loc[best_epoch]
    ax.scatter(
        [best_epoch], [best_value],
        color=VAL_COLOR, edgecolor="white", s=90, zorder=5,
        label=f"Best val. loss (epoch {best_epoch}: {best_value:.3f})"
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3)

    return counts


def plot_training_curves():
    summary_df = pd.read_csv(RESULTS_DIR / "summary.csv")
    
    fractions = [
        (0.25, "25%"),
        (1.00, "100%")
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 7.5),
        sharey=True,
        constrained_layout=True
    )

    for ax, (fraction, label) in zip(axes, fractions):
        history_df = load_histories(round(fraction * 100))
        counts = draw_loss_panel(ax, history_df)

        ax.set_title(
            f"{label} of training data ({format_n_runs(counts)})",
            fontsize=14,
            pad=10
        )

        metrics = calculate_summary(summary_df, fraction)
        ax.text(
            0.5, -0.13,
            build_metrics_text(metrics),
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10,
            color="dimgray"
        )

    axes[0].set_ylabel("Cross-entropy loss", fontsize=12)

    fig.suptitle(
        "Average training and validation loss (shaded area = ±1 std)",
        fontsize=16
    )

    output_path = PLOTS_DIR / "training_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Training curves saved: {output_path}")


def main():
    PLOTS_DIR.mkdir(parents = True, exist_ok = True)

    plot_training_curves()

    csv_files = sorted(RESULTS_DIR.glob("*_confusion_matrix.csv"))

    if not csv_files:
        print("No confusion matrix CSV files found.")
        return

    for csv_path in csv_files:
        plot_confusion_matrix(csv_path)

    print(f"\nGenerated {len(csv_files)} confusion matrices.")


if __name__ == "__main__":
    main()