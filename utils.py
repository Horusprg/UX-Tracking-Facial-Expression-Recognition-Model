import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import numpy
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import KFold

# Configurações de transformação de dados
def get_transforms(img_height, img_width):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(
                size=(img_height, img_width), scale=(0.9, 1.0)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


# Carregamento dos dados
def load_data(data_dir, img_height, img_width, batch_size):
    train_transform, _ = get_transforms(img_height, img_width)

    dataset = datasets.ImageFolder(data_dir)
    dataset.transform = train_transform  # Aplica transformações de treinamento
    return dataset

# Definição do modelo
def model_select(name, feature_extracting=False):
    if name == "EfficientNet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Add classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 8),
        )

    elif name == "Resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Add classifier
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 8),
        )
    elif name == "MobileNet":
        model = models.MobileNetV2()
        # Add classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 8),
        )
    else:
        raise Exception("Model not implemented!")

    # Descongelar camadas
    if not feature_extracting:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


# Função de treinamento e validação
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr=1e-04,
    weight_decay=1e-5,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_model_wts = copy.deepcopy(model.state_dict())

    # Inicializa as variáveis para Early Stopping
    best_f1_score = 0

    # Historic
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],  # Nova métrica
    }

    for epoch in range(epochs):
        # Treinamento
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []

        # Validation evaluation
        val_loss, accuracy, precision, recall, f1 = model_eval(
            criterion=criterion,
            val_loss=val_loss,
            val_loader=val_loader,
            all_labels=all_labels,
            all_preds=all_preds,
            model=model,
            device=device,
        )

        # Atualiza o dicionário de métricas
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(running_loss / len(train_loader))
        metrics["val_loss"].append(val_loss / len(val_loader))
        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}\n"
            f"Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {accuracy}%, "
            f"Val Precision: {precision}, Val Recall: {recall}, Val F1-Score: {f1}"
        )

    
    return metrics, best_model_wts, best_f1_score

def model_eval(criterion, val_loss, val_loader, all_labels, all_preds, model, device):
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = (
        100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    )
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Class Report\n{class_report}")

    return (
        val_loss,
        accuracy,
        precision,
        recall,
        f1
    )
