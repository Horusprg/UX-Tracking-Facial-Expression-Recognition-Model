import os
import time
import torch
from torchvision import transforms
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np

from utils import model_select, load_data, set_seed

# Função para carregar a imagem e transformá-la para o formato que o modelo espera
def load_image(data_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(data_dir, transform)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return dataloader, idx_to_class


def metrics_eval(true_labels, pred_labels):
    accuracy = (
        sum([p == l for p, l in zip(pred_labels, true_labels)]) / len(true_labels)
    )
    precision = precision_score(true_labels, pred_labels, average="macro")
    recall = recall_score(true_labels, pred_labels, average="macro")
    f1 = f1_score(true_labels, pred_labels, average="macro")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    class_report = classification_report(true_labels, pred_labels)

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Class Report\n{class_report}")

    return (accuracy, precision, recall, f1)


if __name__ == "__main__":
    np.int = int
    set_seed(42)
    DATA_DIR = "AffectNet"
    IMG_HEIGHT, IMG_WIDTH = 96, 96
    BATCH_SIZE = 128

    MODEL_NAME1 = "EfficientNet"  # EfficientNet, DaViT, VGG16, Resnet50 or MobileNet
    MODEL_PATH1 = fr"results\best_{MODEL_NAME1}.pth"
    MODEL_NAME2 = "MobileNet"  # EfficientNet, DaViT, VGG16, Resnet50 or MobileNet
    MODEL_PATH2 = fr"results\best_{MODEL_NAME2}.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model1 = model_select(MODEL_NAME1)
    model1.load_state_dict(torch.load(MODEL_PATH1))
    model1.to(DEVICE)
    model1.eval()
    
    model2 = model_select(MODEL_NAME2)
    model2.load_state_dict(torch.load(MODEL_PATH2))
    model2.to(DEVICE)
    model2.eval()

    _, test_dataset = load_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH)
    true_labels = []
    pred_labels = []
    idx_to_class = {v: k for k, v in test_dataset.dataset.class_to_idx.items()}
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs_mean = (outputs1 + outputs2) / 2
            _, preds = torch.max(outputs_mean, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    end_time = time.time()

    # IPS
    total_time = end_time - start_time
    total_inferences = len(test_loader.dataset)
    infer_per_second = total_inferences / total_time

    # Params
    params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)

    total_params = params1 + params2

    # Model Size
    file1_size = os.path.getsize(MODEL_PATH1)
    file2_size = os.path.getsize(MODEL_PATH2)
    file_size_bytes = file1_size + file2_size
    size_in_mb = file_size_bytes / (1024 * 1024)

    true_labels = [idx_to_class[label] for label in true_labels]
    pred_labels = [idx_to_class[label] for label in pred_labels]

    # Performance Metrics
    accuracy, precision, recall, f1 = metrics_eval(
        true_labels=true_labels,
        pred_labels=pred_labels,
    )

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    class_report = classification_report(true_labels, pred_labels)
    print(f"\n---------------------------------------\n")
    print(f"Results on Test Set:\n")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"\n---------------------------------------\n")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Class Report\n{class_report}")
    print(f"\n---------------------------------------\n")
    print(f"\nCost Evaluation:\n")
    print(f"Inferences Per Second (IPS): {infer_per_second}")
    print(f"Total Parameters: {total_params}")
    print(f"Model Size (mb): {size_in_mb}")
