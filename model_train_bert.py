import os
os.environ["USE_TF"] = "0"  # TensorFlow deaktivieren
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_energy_bert(csv_path="bert_prompts_energy.csv"):
    print("Lade CSV...")
    df = pd.read_csv(csv_path)

    prompts = df["Prompt"].astype(str).tolist()
    y = df["Energieverbrauch"].values

    print("Lade SentenceTransformer [paraphrase-multilingual-MiniLM-L12-v2]...")
    bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print("Berechne Embeddings...")
    X = bert_model.encode(prompts, convert_to_numpy=True, show_progress_bar=True)

    print("Trainiere RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    print("Speichere Regressor...")
    joblib.dump(rf, "rf_energy_embed.joblib")
    print("Modell gespeichert: rf_energy_embed.joblib")

    print("Speichere BERT-Embedding-Modell...")
    bert_model.save("bert_embed_model")
    print("BERT-Embedding-Modell gespeichert: bert_embed_model")

if __name__ == "__main__":
    train_energy_bert()
