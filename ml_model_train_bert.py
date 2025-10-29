import os
os.environ["USE_TF"] = "0"  # kein TensorFlow
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

# CSV laden
df = pd.read_csv("bert_prompts_energy.csv")

# Prompts + Zielspalte
prompts = df["Prompt"].tolist()
y = df["Energieverbrauch"].values

# Embedding-Modell
bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Prompt-Embeddings
X = bert_model.encode(prompts, convert_to_numpy=True, show_progress_bar=True)

# RandomForest-Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Modell safe
joblib.dump(rf, "rf_energy_embed.joblib")
print("Modell gespeichert rf_energy_embed.joblib")

# Embedding-Modell safe
bert_model.save("bert_embed_model")

