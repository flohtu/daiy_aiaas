import matplotlib.pyplot as plt
import pandas as pd
import glob

# Liste der CSV-Dateien
csv_dateien = glob.glob("emissions_*.csv")

# Listen für die Daten initialisieren
modelle = []
dauer = []
emissionen = []
emissionsrate = []
cpu_leistung = []
gpu_leistung = []
ram_leistung = []
cpu_energie = []
gpu_energie = []
ram_energie = []
energieverbrauch = []

# Daten aus jeder CSV
for datei in csv_dateien:
    df = pd.read_csv(datei)
    modell_name = datei.split('_')[1].split('.')[0]
    modelle.append(modell_name)
    dauer.append(df['duration'].mean())
    emissionen.append(df['emissions'].mean())
    emissionsrate.append(df['emissions_rate'].mean())
    cpu_leistung.append(df['cpu_power'].mean())
    gpu_leistung.append(df['gpu_power'].mean())
    ram_leistung.append(df['ram_power'].mean())
    cpu_energie.append(df['cpu_energy'].mean())
    gpu_energie.append(df['gpu_energy'].mean())
    ram_energie.append(df['ram_energy'].mean())
    energieverbrauch.append(df['energy_consumed'].mean())

# Subplots erstellen
fig, axs = plt.subplots(5, 2, figsize=(10, 14))
fig.suptitle('Vergleich der Modelle: Energiemetriken', fontsize=12)

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

# GPU-Leistung (W)
axs[2, 0].bar(modelle, gpu_leistung, color='#666666')
axs[2, 0].set_title('GPU-Leistung', fontsize=8)
axs[2, 0].set_ylabel('Watt', fontsize=7)
axs[2, 0].grid(True, axis='y')
axs[2, 0].tick_params(axis='x', rotation=0, labelsize=6)

# RAM-Leistung (W)
axs[2, 1].bar(modelle, ram_leistung, color='#666666')
axs[2, 1].set_title('RAM-Leistung', fontsize=8)
axs[2, 1].set_ylabel('Watt', fontsize=7)
axs[2, 1].grid(True, axis='y')
axs[2, 1].tick_params(axis='x', rotation=0, labelsize=6)

# CPU-Energie (kWh)
axs[3, 0].bar(modelle, cpu_energie, color='#666666')
axs[3, 0].set_title('CPU-Energie', fontsize=8)
axs[3, 0].set_ylabel('kWh', fontsize=7)
axs[3, 0].grid(True, axis='y')
axs[3, 0].tick_params(axis='x', rotation=0, labelsize=6)

# GPU-Energie (kWh)
axs[3, 1].bar(modelle, gpu_energie, color='#666666')
axs[3, 1].set_title('GPU-Energie', fontsize=8)
axs[3, 1].set_ylabel('kWh', fontsize=7)
axs[3, 1].grid(True, axis='y')
axs[3, 1].tick_params(axis='x', rotation=0, labelsize=6)

# RAM-Energie (kWh)
axs[4, 0].bar(modelle, ram_energie, color='#666666')
axs[4, 0].set_title('RAM-Energie', fontsize=8)
axs[4, 0].set_ylabel('kWh', fontsize=7)
axs[4, 0].grid(True, axis='y')
axs[4, 0].tick_params(axis='x', rotation=0, labelsize=6)

# Gesamtenergieverbrauch (kWh)
axs[4, 1].bar(modelle, energieverbrauch, color='#666666')
axs[4, 1].set_title('Gesamtenergieverbrauch', fontsize=8)
axs[4, 1].set_ylabel('kWh', fontsize=7)
axs[4, 1].grid(True, axis='y')
axs[4, 1].tick_params(axis='x', rotation=0, labelsize=6)


plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot anzeigen
plt.show()