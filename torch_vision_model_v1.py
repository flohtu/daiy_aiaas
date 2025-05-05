import torch
from torchvision import models, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Gewichte
weights = models.ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# EuroSAT-Daten
train_dataset = datasets.EuroSAT(root="data/", download=True, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Modell laden
model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 10)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lernrate lr (Schrittgröße) beim Anpassen der Gewichte

# Training (1 Epoche hier)
num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Modell speichern
torch.save(model.state_dict(), "eurosat_resnet50.pth")
print("Modell gespeichert: 'eurosat_resnet50.pth'")
