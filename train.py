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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from utils import load_data, model_select, train_model, set_seed
from eval import metrics_eval


def update_metrics(existing_metrics, new_metrics):
    for key in existing_metrics:
        for value in new_metrics[key]:
            existing_metrics[key].append(value)

def run_kfold_cross_validation(model_name, k=5):
    # Carrega o dataset completo
    train_dataset, test_dataset = load_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH)
    train_labels = [label for _, label in train_dataset]
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    fold_results = []
    model = model_select(model_name)
    model = model.to(DEVICE)
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

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)), train_labels)):
        print(f"\n\nTraining on fold {fold+1}/{k}...")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        new_metrics, model_wts, f1_score = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=EPOCHS,
            lr=LR,
        )
        torch.cuda.empty_cache()
        new_metrics["fold"] = [fold+1]*len(new_metrics["epoch"])

        update_metrics(metrics, new_metrics)

        if f1_score > best_f1_score:
            best_f1_score = f1_score
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
        try:
            for i in range(len(metrics["fold"])):
                row = [metrics[key][i] for key in headers]
                writer.writerow(row)
        except:
            None
    
    # Trained Model Evaluation on Test Set
    final_model = model_select(MODEL_NAME)
    final_model.load_state_dict(torch.load(f"best_{MODEL_NAME}.pth"))
    final_model.to(DEVICE)
    final_model.eval()

    true_labels = []
    pred_labels = []
    idx_to_class = {v: k for k, v in test_dataset.dataset.class_to_idx.items()}
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = final_model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    true_labels = [idx_to_class[label] for label in true_labels]
    pred_labels = [idx_to_class[label] for label in pred_labels]

    accuracy, precision, recall, f1 = metrics_eval(
        true_labels=true_labels,
        pred_labels=pred_labels,
    )

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    class_report = classification_report(true_labels, pred_labels)
    print(f"\n---------------------------------------\n")
    print(f"\nResults on Test Set:\n")
    print(f"Accuracy: {accuracy}%, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")
    print(f"Class Report\n{class_report}")


# Configurações iniciais
if __name__ == "__main__":
    np.int = int
    set_seed(42)
    DATA_DIR = "AffectNet"
    IMG_HEIGHT, IMG_WIDTH = 96, 96
    EPOCHS = 5  # Epochs per fold
    MODEL_NAME = "EfficientNet"  # EfficientNet, Resnet50, MobileNet or VGG16
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LR = 0.0010108536517790942
    BATCH_SIZE = 128

    run_kfold_cross_validation(MODEL_NAME, k=9)
