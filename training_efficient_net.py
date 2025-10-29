import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker
from PIL import Image

print("Starte Setup...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Letterbox Transform (behält ganzes Bild, Seitenverhältnis bleibt)
def letterbox(im, new_size=224, color=(0,0,0)):
    w, h = im.size
    scale = new_size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    im_resized = im.resize((nw, nh), Image.BILINEAR)
    new_im = Image.new('RGB', (new_size, new_size), color)
    new_im.paste(im_resized, ((new_size - nw)//2, (new_size - nh)//2))
    return new_im

class LetterboxTransform:
    def __init__(self, size=224):
        self.size = size
    def __call__(self, img):
        return letterbox(img, new_size=self.size)

# ---- Offizielle ImageNet-Normalisierung ----
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

letterbox_transform = LetterboxTransform(224)
normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

preprocess = transforms.Compose([
    letterbox_transform,
    normalize_transform
])

print("Lade Datasets...")

train_dir = "damaged_packages"
full_dataset = datasets.ImageFolder(root=train_dir, transform=preprocess)

train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

print(f"Gefundene Klassen: {full_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

print("Lade und passe EfficientNet an...")

weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(full_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

tracker = EmissionsTracker(output_file="emissions_efficientnet.csv")
tracker.start()

print("Starte Training (10 Epochen)...")

num_epochs = 10
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

torch.save(model.state_dict(), "damaged_packages_efficientnet_epoch.pth")
print("Modell gespeichert: 'damaged_packages_efficientnet_epoch.pth'")
