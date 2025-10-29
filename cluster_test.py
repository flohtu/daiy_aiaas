import pandas as pd
import requests
from codecarbon import EmissionsTracker

# CONFIG
OLLAMA_API_URL = "http://localhost:11434/api/generate"
PROMPTS_CSV = "prompts_to_measure_energy.csv"   # Eingabe
RESULTS_CSV = "cluster_energy_results.csv"     # Ausgabe

def measure_prompt_energy(prompt, tracker):
    tracker.start()
    try:
        full_prompt = (
            "Du bist ein hilfreicher, deutschsprachiger KI-Chatassistent. "
            "Beantworte die folgende Frage so gut du kannst, basierend auf deinem Wissen.\n\n"
            f"Frage:\n{prompt}\n"
            "Antwort:"
        )

        payload = {"model": "gemma3:1b", "prompt": full_prompt, "stream": False}
        r = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        r.raise_for_status()
        emissions = tracker.stop()
        return emissions

    except Exception as e:
        emissions = tracker.stop()
        print(f"Fehler für Prompt: {prompt}\n{e}")
        return None

def main():
    df = pd.read_csv(PROMPTS_CSV)
    prompts = df["Prompt"].astype(str).tolist()
    results = []

    for prompt in prompts:
        print(f"Bearbeite: {prompt[:60]}...")
        tracker = EmissionsTracker(project_name="cluster_test", measure_power_secs=5)
        emissions = measure_prompt_energy(prompt, tracker)
        results.append({
            "Prompt": prompt,
            "EnergyMeasured": emissions,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULTS_CSV, index=False)
    print(f"Messungen abgeschlossen. Ergebnisse: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
