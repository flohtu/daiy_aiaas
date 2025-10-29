import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# CSV laden
df = pd.read_csv("chatbot_energieverbrauch_v3.csv")

# Features & Zielspalte
feature_cols = [
    "Prompt_Laenge",
    "Prompt_Woerter",
    "Enthaelt_Ausfuehrlich_Schluesselwort",
    "Enthaelt_Kurz_Schluesselwort",
    "Anzahl_Sonderzeichen",
    "Anzahl_Zahlen"
]
X = df[feature_cols]
y = df["Energieverbrauch"]

# training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# abspeichern
joblib.dump(model, "rf_energy_model.joblib")
print("Fertig! Modell als rf_energy_model.joblib gespeichert.")
