import copy
import time
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

from .config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    WEIGHT_DECAY
)

from .utils import autocast_context, create_grad_scaler


def run_epoch(model, loader, criterion, device, optimizer = None, scaler = None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = torch.zeros((), dtype = torch.float64, device = device)
    all_predictions = []
    all_labels = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
                optimizer.zero_grad(set_to_none = True)

            with autocast_context(device):
                outputs = model(images)

            loss = criterion(outputs.float(), labels)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.detach().double() * labels.size(0)

            all_predictions.append(outputs.argmax(dim = 1))
            all_labels.append(labels)

    average_loss = total_loss.item() / loader.num_samples

    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    accuracy = accuracy_score(all_labels, all_predictions)

    f1 = f1_score(
        all_labels,
        all_predictions,
        average = "macro",
        zero_division = 0
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "f1_macro": f1
    }


def train_model(
    model,
    train_loader,
    val_loader,
    device = DEVICE,
    epochs = EPOCHS,
    learning_rate = LEARNING_RATE,
    patience = PATIENCE,
    weight_decay = WEIGHT_DECAY,
    checkpoint_path = None
):
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    if patience < 1:
        raise ValueError("Patience must be at least 1.")
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay
    )

    scaler = create_grad_scaler(device)

    history = []

    best_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"]
        })

        print(
            f"Epoch [{epoch + 1}] "
            f"| TL: {train_metrics['loss']:.4f} "
            f"| TACC: {train_metrics['accuracy']:.4f} "
            f"| TF1: {train_metrics['f1_macro']:.4f} "
            f"| VL: {val_metrics['loss']:.4f} "
            f"| VACC: {val_metrics['accuracy']:.4f} "
            f"| VF1: {val_metrics['f1_macro']:.4f} "
        )

        if val_metrics["loss"] < best_loss:            
            best_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                torch.save({
                    "model_state_dict": best_model_state,
                    "best_val_loss": best_loss,
                    "epoch": best_epoch
                }, checkpoint_path)

            print("[!] New best model.")
        else :
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[!] Early stopping at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state)

    return history, best_epoch


def benchmark_training(
    model_factory,
    train_loader,
    val_loader,
    device = DEVICE,
    learning_rate = LEARNING_RATE,
    weight_decay = WEIGHT_DECAY,
    warmup_epochs = 1,
    measured_epochs = 3
):
    if warmup_epochs < 0:
        raise ValueError("Warm-up epochs cannot be negative.")

    if measured_epochs < 1:
        raise ValueError("Measured epochs must be at least 1.")

    print(f"    Device: {device} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    model = model_factory().to(device, memory_format = torch.contiguous_format)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay
    )

    scaler = create_grad_scaler(device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)


    #   Warm-up
    for i in range(warmup_epochs):
        print(f"    Warm-up {i + 1}/{warmup_epochs}...", end = " ", flush = True)
        start_time = time.perf_counter()

        run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler
        )

        run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        synchronize()
        print(f"done in {time.perf_counter() - start_time:.1f}s")

    epoch_times = []


    #   Measuring
    for i in range(measured_epochs):
        print(f"    Measuring {i + 1}/{measured_epochs}...", end = " ", flush = True)

        synchronize()
        start_time = time.perf_counter()

        run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler
        )

        run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        synchronize()
        elapsed = time.perf_counter() - start_time

        epoch_times.append(elapsed)
        print(f"{elapsed:.1f}s")

    average_epoch_time = sum(epoch_times) / len(epoch_times)

    return average_epoch_time