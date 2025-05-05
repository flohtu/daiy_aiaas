import torch
from torchvision import models
from PIL import Image

# Modell laden
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("eurosat_resnet50.pth"))
model.eval()

# Gewichte
weights = models.ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# Bild (Eingabe)
image_path = ""
image = Image.open(image_path).convert("RGB")
image = preprocess(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

classes = [
    'Ackerland',               # AnnualCrop
    'Wald',                    # Forest
    'Krautige Vegetation',     # HerbaceousVegetation
    'Autobahn',                # Highway
    'Industriegebiet',         # Industrial
    'Weideland',               # Pasture
    'Dauerkulturen',           # PermanentCrop (z.B. Weinberge)
    'Wohngebiet',              # Residential
    'Fluss',                   # River
    'Meer/See'                 # SeaLake
]

print(f"Prediction: {classes[predicted.item()]}")
