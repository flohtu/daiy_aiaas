import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker


print("Starte Setup...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

weights = models.ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

print("Lade Datasets...")

train_dir = "damaged_packages"
full_dataset = datasets.ImageFolder(root=train_dir, transform=preprocess)

train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

print(f"Gefundene Klassen: {full_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

print("Lade und passe ResNet18 an...")

model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

tracker = EmissionsTracker(output_file="emissions_resnet18.csv")

tracker.start()

print("Starte Training (5 Epochen)...")

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

    epoch_loss = running_loss / len(train_loader)
    print(f"Trainings-Loss: {epoch_loss:.4f}")

tracker.stop()

torch.save(model.state_dict(), "damaged_packages_resnet18_epoch.pth")
print("Modell gespeichert: 'damaged_packages_resnet18_epoch.pth'")
