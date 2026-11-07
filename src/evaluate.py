import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .config import DEVICE


def evaluate_model(
    model,
    test_loader,
    class_names,
    device = DEVICE
):
    if len(class_names) != len(test_loader.dataset.dataset.classes):
        raise ValueError("Class names do not match the dataset classes")
    
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim = 1)

            total_loss += loss.item() * labels.size(0)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(test_loader.dataset)

    accuracy = accuracy_score(all_labels, all_predictions)

    f1_macro = f1_score(
        all_labels,
        all_predictions,
        average = "macro",
        zero_division = 0
    )

    num_classes = len(class_names)

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels = list(range(num_classes))
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels = list(range(num_classes)),
        target_names = class_names,
        output_dict = True,
        zero_division = 0
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": matrix,
        "classification_report": report
    }