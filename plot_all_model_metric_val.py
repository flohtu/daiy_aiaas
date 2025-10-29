import random
import torch
from torchvision import models, datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
import matplotlib.pyplot as plt

# Seed setzen
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # falls Multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# GPU oder CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Dataset
train_dir = "damaged_packages"

# Modelle
model_names = ["alexnet", "efficientnet", "mobilenetv3", "resnet18", "resnet50", "vgg16", "densenet"]

# Dictionary für Metriken
all_metrics = {}

# Für jedes Modell - Validierung
for model_name in model_names:
    print(f"\nValidierung für Modell: {model_name}")

    # Initialisierung und Gewichte
    if model_name == "alexnet":
        weights = models.AlexNet_Weights.DEFAULT
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif model_name == "efficientnet":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "mobilenetv3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif model_name == "densenet":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)

    # Transformationen laden
    preprocess = weights.transforms()

    # Dataset laden
    full_dataset = datasets.ImageFolder(root=train_dir, transform=preprocess)

    # 80% Training und 20% Validierung
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    # Validierungs-Loader
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    # Klassen (aus dem Datensatz)
    classes = full_dataset.classes
    print(f"Klassen: {classes}")

    # Modell laden
    model_path = f"damaged_packages_{model_name}_epoch.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Loss-Funktion
    criterion = nn.CrossEntropyLoss()

    # Validierung
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metriken berechnen
    val_loss = val_loss / len(valid_loader)
    val_accuracy = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Metriken speichern
    metrics = {
        "val_loss": round(val_loss, 4),
        "val_accuracy": round(val_accuracy, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": conf_matrix.tolist()
    }

    all_metrics[model_name] = metrics

    # Ausgabe
    print(f"Validierungs-Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

# Metriken in JSON speichern
with open("metrics_all_models.json", "w") as f:
    json.dump(all_metrics, f)

# Grafische Darstellung der Accuracy
plt.figure(figsize=(8, 6))
plt.suptitle('Vergleich der Modelle: Accuracy', fontsize=12)

# Liste für Accuracy
val_accuracies = [all_metrics[model]["val_accuracy"] for model in model_names]

# Accuracy-Plot
plt.bar(model_names, val_accuracies, color='#666666')
plt.title('Accuracy', fontsize=10)
plt.ylabel('%', fontsize=8)
plt.grid(True, axis='y')
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()

# Plot anzeigen
plt.show()