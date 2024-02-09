import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from skopt.utils import use_named_args
from torch.utils.data import DataLoader, random_split
from skopt.space import Real, Integer
from skopt import gp_minimize
import numpy as np
import csv
from sklearn.metrics import f1_score

from utils import model_select, get_transforms, set_seed

# Hyperparameters Space
space = [
    Real(1e-8, 1e-2, "log-uniform", name="learning_rate"),
    Integer(128, 512, name="batch_size"),
]

def load_split_data(data_dir, img_height, img_width, batch_size):
    train_transform, val_transform = get_transforms(img_height, img_width)

    dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

@use_named_args(space)
def objective(learning_rate, batch_size):
    # Configurações principais
    batch_size = int(batch_size)

    train_loader, val_loader = load_split_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, batch_size)
    
    criterion = nn.CrossEntropyLoss()
    model = model_select(MODEL_NAME) 
    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_f1_score = train_and_evaluate(
        model, train_loader, val_loader, optimizer, criterion, EPOCHS, DEVICE
    )
    print(f'LR: {learning_rate}, Batch Size: {batch_size}, F1-Score Achieved: {best_f1_score}')
    # Armazene os resultados em um arquivo CSV
    with open(OUTPUT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([learning_rate, batch_size, best_f1_score])
    return -best_f1_score


def train_and_evaluate(
    model, train_loader, val_loader, optimizer, criterion, epochs, device
):
    best_f1_score = 0
    model.to(device)

    for epoch in range(epochs):
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

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        best_f1_score = max(best_f1_score, f1)

        print(
            f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, F1-Score: {f1}%"
        )

    return best_f1_score


if __name__ == '__main__':
    np.int = int
    set_seed(42)
    DATA_DIR = 'AffectNet'
    IMG_HEIGHT, IMG_WIDTH = 96, 96
    EPOCHS = 5 # Epochs per optimization candidate
    MODEL_NAME = "DaViT"  # DaViT, EfficientNet, Resnet50, MobileNet, SENet50 or VGG16
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CALLS = 25
    OUTPUT_FILE = f'{MODEL_NAME}_optimization_results.csv'
    # Inicialize o arquivo CSV
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Learning Rate', 'Batch Size', 'F1-Score'])
    result = gp_minimize(objective, space, n_calls=CALLS, random_state=0)
    print(f"\n\nResultados Obtidos para a otimização de hiperparâmetros do {MODEL_NAME}\nMelhor learning_rate: {result.x[0]}\nMelhor Batch Size: {result.x[1]}\n")