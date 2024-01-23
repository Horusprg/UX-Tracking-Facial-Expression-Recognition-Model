import torch
import torch.nn as nn
from torch.optim import Adam
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from skopt import gp_minimize
import numpy as np
import csv
from sklearn.metrics import f1_score

from utils import load_data, model_select

# Hyperparameters Space
space = [
    Real(1e-8, 1e-2, "log-uniform", name="learning_rate"),
    Integer(128, 512, name="batch_size"),
]


@use_named_args(space)
def objective(learning_rate, batch_size):
    # Configurações principais
    batch_size = int(batch_size)

    train_loader, val_loader = load_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, batch_size)
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
    DATA_DIR = 'AffectNet'
    IMG_HEIGHT, IMG_WIDTH = 96, 96
    EPOCHS = 5 # Epochs per optimization candidate
    MODEL_NAME = 'EfficientNet' # EfficientNet, Resnet50 or MobileNet
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CALLS = 25
    OUTPUT_FILE = f'{MODEL_NAME}_optimization_results.csv'
    # Inicialize o arquivo CSV
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Learning Rate', 'Batch Size', 'F1-Score'])
    result = gp_minimize(objective, space, n_calls=CALLS, random_state=0)
    print(f"\n\nResultados Obtidos para a otimização de hiperparâmetros do {MODEL_NAME}\nMelhor learning_rate: {result.x[0]}\nMelhor Batch Size: {result.x[1]}\n")

    
# Resnet50 Optm ->
# EfficientNet Optm -> Melhor learning_rate: 0.0013178605614389228, Melhor Batch Size: 64