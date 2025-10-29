import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Liste der CSV-Dateien
csv_dateien = glob.glob("emissions_*.csv")

# Listen für die Daten initialisieren
modelle = []
dauer = []
emissionen = []
emissionsrate = []
cpu_leistung = []
ram_leistung = []
cpu_energie = []
ram_energie = []
energieverbrauch = []
modellgroesse = []

# Daten aus jeder CSV und .pth-Datei
for datei in csv_dateien:
    df = pd.read_csv(datei)
    modell_name = datei.split('_')[1].split('.')[0]
    modelle.append(modell_name)
    dauer.append(df['duration'].mean())
    emissionen.append(df['emissions'].mean())
    emissionsrate.append(df['emissions_rate'].mean())
    cpu_leistung.append(df['cpu_power'].mean())
    ram_leistung.append(df['ram_power'].mean())
    cpu_energie.append(df['cpu_energy'].mean())
    ram_energie.append(df['ram_energy'].mean())
    energieverbrauch.append(df['energy_consumed'].mean())

    # Calculate model size (in MB)
    pth_file = f"damaged_packages_{modell_name}_epoch.pth"
    if os.path.exists(pth_file):
        size_mb = os.path.getsize(pth_file) / (1024 * 1024)  # Convert bytes to MB
        modellgroesse.append(size_mb)
    else:
        modellgroesse.append(0)
        print(f"Warning: {pth_file} not found")

# Subplots erstellen (5x2: 9 Plots + 1 leer)
fig, axs = plt.subplots(5, 2, figsize=(10, 14))

# Dauer (Sekunden)
axs[0, 0].bar(modelle, dauer, color='#666666')
axs[0, 0].set_title('Dauer', fontsize=8)
axs[0, 0].set_ylabel('Sekunden', fontsize=7)
axs[0, 0].grid(True, axis='y')
axs[0, 0].tick_params(axis='x', rotation=0, labelsize=6)

# Emissionen
axs[0, 1].bar(modelle, emissionen, color='#666666')
axs[0, 1].set_title('Emissionen', fontsize=8)
axs[0, 1].set_ylabel('kg CO₂', fontsize=7)
axs[0, 1].grid(True, axis='y')
axs[0, 1].tick_params(axis='x', rotation=0, labelsize=6)

# Emissionsrate (kg/kWh)
axs[1, 0].bar(modelle, emissionsrate, color='#666666')
axs[1, 0].set_title('Emissionsrate', fontsize=8)
axs[1, 0].set_ylabel('kg CO₂/kWh', fontsize=7)
axs[1, 0].grid(True, axis='y')
axs[1, 0].tick_params(axis='x', rotation=0, labelsize=6)

# CPU-Leistung (W)
axs[1, 1].bar(modelle, cpu_leistung, color='#666666')
axs[1, 1].set_title('CPU-Leistung', fontsize=8)
axs[1, 1].set_ylabel('Watt', fontsize=7)
axs[1, 1].grid(True, axis='y')
axs[1, 1].tick_params(axis='x', rotation=0, labelsize=6)

# RAM-Leistung (W)
axs[2, 0].bar(modelle, ram_leistung, color='#666666')
axs[2, 0].set_title('RAM-Leistung', fontsize=8)
axs[2, 0].set_ylabel('Watt', fontsize=7)
axs[2, 0].grid(True, axis='y')
axs[2, 0].tick_params(axis='x', rotation=0, labelsize=6)

# CPU-Energie (kWh)
axs[2, 1].bar(modelle, cpu_energie, color='#666666')
axs[2, 1].set_title('CPU-Energie', fontsize=8)
axs[2, 1].set_ylabel('kWh', fontsize=7)
axs[2, 1].grid(True, axis='y')
axs[2, 1].tick_params(axis='x', rotation=0, labelsize=6)

# RAM-Energie (kWh)
axs[3, 0].bar(modelle, ram_energie, color='#666666')
axs[3, 0].set_title('RAM-Energie', fontsize=8)
axs[3, 0].set_ylabel('kWh', fontsize=7)
axs[3, 0].grid(True, axis='y')
axs[3, 0].tick_params(axis='x', rotation=0, labelsize=6)

# Gesamtenergieverbrauch (kWh)
axs[3, 1].bar(modelle, energieverbrauch, color='#666666')
axs[3, 1].set_title('Gesamtenergieverbrauch', fontsize=8)
axs[3, 1].set_ylabel('kWh', fontsize=7)
axs[3, 1].grid(True, axis='y')
axs[3, 1].tick_params(axis='x', rotation=0, labelsize=6)

# Modellgröße (MB)
axs[4, 0].bar(modelle, modellgroesse, color='#666666')
axs[4, 0].set_title('Modellgröße', fontsize=8)
axs[4, 0].set_ylabel('Megabytes', fontsize=7)
axs[4, 0].grid(True, axis='y')
axs[4, 0].tick_params(axis='x', rotation=0, labelsize=6)

# Empty subplot (to maintain grid)
axs[4, 1].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig('model_metrics_no_gpu.png')
