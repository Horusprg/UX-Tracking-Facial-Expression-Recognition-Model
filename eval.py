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
        100 * sum([p == l for p, l in zip(pred_labels, true_labels)]) / len(true_labels)
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
    MODEL_NAME = "EfficientNet"  # EfficientNet, Resnet50 or MobileNet
    MODEL_PATH = "results/best_EfficientNet.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_select(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    train_dataset, test_dataset = load_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH)
    true_labels = []
    pred_labels = []
    idx_to_class = {v: k for k, v in test_dataset.dataset.class_to_idx.items()}
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
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