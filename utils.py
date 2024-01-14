import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import precision_score, recall_score

# Configurações de transformação de dados
def get_transforms(img_height, img_width):
    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size=(img_height, img_width), scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

# Carregamento dos dados
def load_data(data_dir, img_height, img_width, batch_size):
    train_transform, val_transform = get_transforms(img_height, img_width)

    dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

# Definição do modelo
def model_select(img_height, img_width):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 8)  # 8 classes
    return model

# Função de treinamento e validação
def train_model(model, train_loader, val_loader, device, epochs, metrics_path='training_metrics.txt', patience=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.1)

    # Inicializa as variáveis para Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Inicializa um dicionário para armazenar as métricas
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'accuracy': [], 'precision': [], 'recall': []}

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
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        # Atualiza o dicionário de métricas
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(running_loss / len(train_loader))
        metrics['val_loss'].append(val_loss / len(val_loader))
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)

        scheduler.step()

        # Verifica o Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early Stopping at epoch {epoch+1}')
                early_stop = True
                break

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, '
              f'Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {accuracy}%, '
              f'Val Precision: {precision}, Val Recall: {recall}')

    torch.save(model.state_dict(), 'best_model.pth')

    # Escreve as métricas em um arquivo .txt
    with open(metrics_path, 'w') as f:
        for key in metrics:
            f.write(f"{key}: {metrics[key]}\n")