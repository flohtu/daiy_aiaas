import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker

# https://www.kaggle.com/datasets/christianvorhemus/industrial-quality-control-of-packages/data
print("Starte Setup...")

# Device (GPU oder CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# ResNet50 Weights & Transformation
weights = models.ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

print("Lade Datasets...")

# Dataset Pfad
train_dir = "damaged_packages"

# Erstelle den vollständigen Datensatz
full_dataset = datasets.ImageFolder(root=train_dir, transform=preprocess)

# Aufteilung in 80% Training und 20% Validierung
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

print(f"Gefundene Klassen: {full_dataset.classes}")

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

print("Lade und passe ResNet50 an...")

# ResNet50 Modell anpassen
model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))  # Anzahl der Klassen
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


tracker = EmissionsTracker(output_file="emissions_resnet50.csv")
tracker.start()

print("Starte Training (5 Epochen)...")

# Training — 5 Epochen
num_epochs = 5
for epoch in range(num_epochs):
    print(f"\nEpoche {epoch + 1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}/{len(train_loader)}")

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Trainings-Loss pro Epoche ausgeben
    epoch_loss = running_loss / len(train_loader)
    print(f"Trainings-Loss: {epoch_loss:.4f}")

tracker.stop()

# Speichern
torch.save(model.state_dict(), "damaged_packages_resnet50_epoch.pth")
print("Modell gespeichert: 'damaged_packages_resnet50_epoch.pth'")