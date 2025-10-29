import time
import requests
import pandas as pd
import os
import random
from codecarbon import EmissionsTracker

# Pfad zur Datei mit den Prompts
PROMPT_DATEI = "base_prompts.txt"

# Prompts aus der Datei lesen
with open(PROMPT_DATEI, "r", encoding="utf-8") as f:
    base_prompts = [line.strip() for line in f if line.strip()]

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

def generate_ollama_response(prompt):
    system = (
        "Du bist ein hilfreicher deutschsprachiger Chatbot. "
        "Beantworte die Frage so, wie es der Prompt verlangt."
    )
    data = {
        "model": OLLAMA_MODEL,
        "prompt": f"SYSTEM: {system}\nUSER: {prompt}\n",
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=data)
    response.raise_for_status()
    return response.json()["response"]

output_file = "bert_prompts_energy.csv"
header_needed = not os.path.exists(output_file)

for idx, prompt in enumerate(base_prompts):
    tracker = EmissionsTracker(project_name="ollama-chatbot")
    tracker.start()
    try:
        antwort = generate_ollama_response(prompt)
    except Exception as e:
        print(f"Fehler bei Prompt: {prompt} - {e}")
        tracker.stop()
        continue
    emissions = tracker.stop()

    result = {
        "Prompt": prompt,
        "Energieverbrauch": emissions
    }
    df = pd.DataFrame([result])
    df.to_csv(output_file, mode='a', header=header_needed, index=False, encoding='utf-8')
    header_needed = False

    print(f"Prompt {idx + 1}/{len(base_prompts)} fertig.")
    time.sleep(1)

print("CSV gespeichert!")
