import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .config import DEVICE
from .utils import autocast_context


def evaluate_model(
    model,
    test_loader,
    class_names,
    device = DEVICE
):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = torch.zeros((), dtype = torch.float64, device = device)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            with autocast_context(device):
                outputs = model(images)

            loss = criterion(outputs.float(), labels)

            total_loss += loss.detach().double() * labels.size(0)

            all_predictions.append(outputs.argmax(dim = 1))
            all_labels.append(labels)

    average_loss = total_loss.item() / test_loader.num_samples

    all_predictions = torch.cat(all_predictions).cpu().tolist()
    all_labels = torch.cat(all_labels).cpu().tolist()

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