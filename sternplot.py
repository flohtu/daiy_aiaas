import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

emission_cols = [
    "duration", "emissions", "emissions_rate",
    "cpu_power", "ram_power", "cpu_energy", "ram_energy", "energy_consumed"
]

label_mapping = {
    "accuracy": "Genauigkeit",
    "duration": "Dauer [s]",
    "emissions": "Emissionen [kg CO₂]",
    "emissions_rate": "Emissionsrate [kg CO₂/kWh]",
    "cpu_power": "CPU-Leistung [W]",
    "ram_power": "RAM-Leistung [W]",
    "cpu_energy": "CPU-Energie [kWh]",
    "ram_energy": "RAM-Energie [kWh]",
    "energy_consumed": "Gesamtenergie [kWh]"
}

model_names = [
    "alexnet", "densenet", "efficientnet",
    "mobilenetv3", "resnet18", "resnet50", "vgg16"
]

emission_files = {
    "alexnet": "emissions_alexnet.csv",
    "densenet": "emissions_densenet.csv",
    "efficientnet": "emissions_efficientnet.csv",
    "mobilenetv3": "emissions_mobilenetv3.csv",
    "resnet18": "emissions_resnet18.csv",
    "resnet50": "emissions_resnet50.csv",
    "vgg16": "emissions_vgg16.csv"
}

with open('metrics_all_models.json', 'r') as f:
    metrics = json.load(f)
accuracies = {name: float(metrics[name]['val_accuracy']) / 100 for name in model_names}

emissions_data = {}
for model in model_names:
    df = pd.read_csv(emission_files[model])
    row = df.iloc[0]
    vals = {}
    for col in emission_cols:
        try:
            vals[col] = float(row[col])
        except Exception as e:
            print(f"WARNUNG: Spalte {col} fehlt in {emission_files[model]} oder kann nicht als float gelesen werden.")
            vals[col] = np.nan
    emissions_data[model] = vals

all_metric_names = ["accuracy"] + emission_cols

model_metric_values = {}
for model in model_names:
    vals = [accuracies[model]] + [emissions_data[model][col] for col in emission_cols]
    model_metric_values[model] = vals

def normalize_individual(vals):
    minv = np.nanmin(vals)
    maxv = np.nanmax(vals)
    return [(v - minv) / (maxv - minv) if (maxv > minv) and not np.isnan(v) else 0.5 for v in vals]

def plot_radar(model_name, values, labels, savepath):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = list(values) + [values[0]]
    angles += angles[:1]
    labels_de = [label_mapping.get(l, l) for l in labels]
    labels_de[0] = label_mapping.get("accuracy", "Accuracy")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_de, fontsize=11)
    ax.set_title(f"Radar Plot: {model_name}", fontsize=15)
    ax.grid(True)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

output_dir = "model_radarplots_individual"
os.makedirs(output_dir, exist_ok=True)

for model in model_names:
    vals = model_metric_values[model]
    vals_norm = normalize_individual(vals)
    savepath = os.path.join(output_dir, f"radarplot_{model}_individual.png")
    plot_radar(model, vals_norm, all_metric_names, savepath)

print(f"Alle *individuell normalisierten* Radarplots wurden gespeichert in: {output_dir}")
