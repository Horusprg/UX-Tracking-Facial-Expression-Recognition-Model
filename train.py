"""
All the code was developed by Flávio Moura, a scientific research from the Federal University of Pará and member of the 
Operational Research Laboratory. This repo aims to train multiple models for the task of facial expression recognition.
The main objetive of the development of that models is to be deployed in monitoring tools from the UX-Tracking Framework,
also developed by Flávio Moura. For that reason, the models developed was evaluated focusing on Precision metric and low
computational cost architectures of AI models was selected.

For more informations about:
    - Development of the models
    - Architecture of the UX-Tracking framework
    - Scientif Research behind this code and the fully framework
    - Colaboration in a scientific research

Please, contact Flávio Moura in the email: flavio.moura@itec.ufpa.br
"""

import csv
import torch
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import copy
from sklearn.model_selection import KFold

from utils import load_data, model_select, train_model


def update_metrics(existing_metrics, new_metrics):
    for key in existing_metrics:
        existing_metrics[key].extend(new_metrics[key])

def run_kfold_cross_validation(model_name, k=5):
    # Carrega o dataset completo
    dataset = load_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH)

    kf = KFold(n_splits=k, shuffle=True)
    fold_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1_score = 0

    metrics = {
        "fold": [],
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training on fold {fold+1}/{k}...")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = model_select(model_name)
        model = model.to(DEVICE)

        new_metrics, model_wts, f1_score = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=EPOCHS,
            lr=LR,
        )
        torch.cuda.empty_cache()
        new_metrics["fold"] = fold+1

        update_metrics(metrics, new_metrics)

        if f1_score > best_f1_score:
            best_model_wts = model_wts
        fold_results.append(best_f1_score)

        print(f"Fold {fold+1} completed. Best F1 Score: {best_f1_score}")

    torch.save(best_model_wts, f"best_{MODEL_NAME}.pth")

    metrics_path = f"{MODEL_NAME}_metrics.csv"

    with open(metrics_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        headers = [
            "fold",
            "epoch",
            "train_loss",
            "val_loss",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
        writer.writerow(headers)

        for i in range(2 * len(metrics["fold"])):
            row = [metrics[key][i] for key in headers]
            writer.writerow(row)

    avg_f1_score = np.mean(fold_results)
    print(f"Average F1 Score across all folds: {avg_f1_score}")
    print(f"Best F1 Score across all folds: {best_f1_score}")


# Configurações iniciais
if __name__ == "__main__":
    np.int = int
    DATA_DIR = "AffectNet"
    IMG_HEIGHT, IMG_WIDTH = 96, 96
    EPOCHS = 1  # Epochs per fold
    MODEL_NAME = "Resnet50"  # EfficientNet, Resnet50 ou MobileNet
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LR = 0.0013178605614389228
    BATCH_SIZE = 64

    run_kfold_cross_validation(MODEL_NAME, k=10)
