import requests
import pandas as pd
import re
from codecarbon import EmissionsTracker
import time
import os
import random

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

def generate_ollama_response(prompt):
    system = ("Du bist ein hilfreicher deutschsprachiger Chatbot. "
              "Beantworte die Frage so, wie es der Prompt verlangt.")
    data = {
        "model": OLLAMA_MODEL,
        "prompt": f"SYSTEM: {system}\nUSER: {prompt}\n",
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=data)
    response.raise_for_status()
    return response.json()["response"]

def count_words(text):
    return len(re.findall(r"\w+", text))

def has_keyword(text, keyword_list):
    text_lower = text.lower()
    return int(any(k in text_lower for k in keyword_list))

def count_special_chars(text):
    # Zählt alle Zeichen, die KEIN Buchstabe/Zahl/Leerzeichen sind
    return len(re.findall(r"[^\w\s]", text))

def count_numbers(text):
    return len(re.findall(r"\d+", text))

# Schlüsselwortlisten
keywords_ausfuehrlich = [
    "ausführlich",
    "detailliert",
    "erkläre",
    "nennen",
    "beschreibe",
    "in eigenen worten",
    "mit beispiel",
    "umfassend",
    "schritt für schritt",
    "genau",
    "erläutere",
    "ausführliche erklärung",
    "erklären sie",
    "begründe",
    "beantworte ausführlich",
    "erörtere",
    "liefere details",
    "liefere eine genaue beschreibung",
    "mit hintergrundwissen"
]

keywords_kurz = [
    "kurz",
    "knapp",
    "in einem satz",
    "stichpunkte",
    "in wenigen worten",
    "kurz und bündig",
    "so kurz wie möglich",
    "prägnant",
    "kurze antwort",
    "eine zeile",
    "maximal ein satz",
    "kurz zusammengefasst",
    "nur das wichtigste",
    "in drei worten",
    "kurz erklären",
    "kurz beschreiben",
    "in kurzform",
    "stichpunktartig",
    "ohne ausschmückung",
    "so knapp wie möglich"
]

base_prompts = [
    "Was versteht man unter AI as a Service?",
    "Nenne die Hauptvorteile von Machine Learning in der Cloud.",
    "Welche Herausforderungen gibt es bei KI-Ethik?",
    "Wie funktioniert supervised learning?",
    "Was ist der Unterschied zwischen klassischem Programmieren und ML?",
    "Wofür wird Reinforcement Learning eingesetzt?",
    "Was ist ein neuronales Netz?",
    "Erkläre Overfitting mit einem Beispiel.",
    "Wie kann man Daten für ML-Anwendungen vorverarbeiten?",
    "Welche Bedeutung haben GPUs für KI?",
    "Was ist Transfer Learning?",
    "Nenne bekannte Anbieter von AIaaS.",
    "Wie lässt sich KI für die Automatisierung nutzen?",
    "Was sind Biases in ML-Modellen?",
    "Was ist ein Hyperparameter in ML?",
    "Wie funktioniert Feature Engineering?",
    "Was ist die Rolle von APIs im Kontext von AIaaS?",
    "Wie wird Natural Language Processing in der Industrie eingesetzt?",
    "Welche Programmiersprachen sind am wichtigsten für KI?",
    "Erkläre den Begriff Big Data.",
    "Was sind die Risiken von generativer KI?",
    "Wie kann man ML-Modelle überwachen?",
    "Was ist Explainable AI (XAI)?",
    "Wie funktioniert ein Decision Tree?",
    "Erkläre Data Augmentation.",
    "Welche Cloud-Plattformen bieten KI-Dienste?",
    "Wie kann man Deep Learning effizient trainieren?",
    "Was sind Use Cases für KI im Gesundheitswesen?",
    "Wie funktioniert ein Recommendation System?",
    "Was ist unsupervised learning?",
    "Was ist ein Random Forest?",
    "Wie erkennt man Anomalien mit ML?",
    "Welche Rolle spielt Data Privacy bei AIaaS?",
    "Wie funktioniert der Turing-Test?",
    "Erkläre Computer Vision.",
    "Wie kann man Modelle gegen Angriffe absichern?",
    "Was ist ein Pretrained Model?",
    "Wie funktioniert Federated Learning?",
    "Was ist die Rolle von Open Source in KI?",
    "Wie werden ML-Modelle in Produktion gebracht?",
    "Was ist der Unterschied zwischen KI und ML?",
    "Wie kann KI nachhaltiger werden?",
    "Wie funktioniert sentiment analysis?",
    "Wie werden Chatbots trainiert?",
    "Welche Herausforderungen gibt es bei der Datenqualität?",
    "Wie kann man Bias im Training reduzieren?",
    "Was sind die Trends im Bereich AIaaS?",
    "Wie kann man ML-Projekte agil umsetzen?",
    "Was ist ein Datensatz?",
    "Wie wichtig ist Datenqualität für KI-Systeme?",
    "Was ist der Unterschied zwischen künstlicher Intelligenz und schwacher KI?",
    "Wie kann Explainable AI zur Akzeptanz von KI beitragen?",
    "Welche regulatorischen Anforderungen gelten für KI-Systeme in Europa?",
    "Was sind Edge AI und ihre Vorteile?",
    "Wie können Unternehmen von AIaaS profitieren?",
    "Welche ethischen Leitlinien sollten bei KI-Projekten beachtet werden?",
    "Wie wird die Qualität von Trainingsdaten gemessen?",
    "Was sind typische Fehlerquellen bei ML-Projekten?",
    "Wie erfolgt die Evaluation von KI-Modellen?",
    "Was ist eine Konfusionsmatrix?",
    "Wie kann Cloud Computing die Entwicklung von KI beschleunigen?",
    "Wie sieht die Zukunft von AIaaS aus?",
    "Was versteht man unter Model Deployment?",
    "Wie lassen sich KI-Modelle skalieren?",
    "Wie kann man Overfitting verhindern?",
    "Welche Tools gibt es für das Monitoring von ML-Modellen?",
    "Wie wird Datenanonymisierung im ML-Kontext umgesetzt?",
    "Was sind typische Herausforderungen bei der Integration von KI in bestehende Systeme?",
    "Wie kann man Datensätze für ML synthetisch erzeugen?",
    "Wie können Unternehmen KI verantwortungsvoll einsetzen?",
    "Was ist Transfer Learning und wann ist es sinnvoll?",
    "Welche Rolle spielen Datenpipelines in ML-Projekten?",
    "Wie kann man Modelle für verschiedene Sprachen trainieren?",
    "Wie wirkt sich die Größe eines Modells auf dessen Energieverbrauch aus?",
    "Was sind aktuelle Trends in der Forschung zu generativer KI?",
    "Wie kann Bias in Datensätzen erkannt werden?",
    "Wie läuft ein typischer AIaaS-Prozess ab?",
    "Was sind die wichtigsten Anforderungen an Datensicherheit bei AIaaS?",
    "Wie wird Explainable AI in der Praxis angewendet?",
    "Was bedeutet Zero-Shot Learning?",
    "Welche Rolle spielt Transfer Learning im KI-Bereich?",
    "Wie können KI-Modelle in Echtzeit eingesetzt werden?",
    "Was sind die Vorteile von AutoML?",
    "Wie wird Fairness in KI-Systemen sichergestellt?",
    "Welche KI-Technologien werden in der Industrie am meisten genutzt?",
    "Was sind die Aufgaben eines Data Scientists?",
    "Welche Unterschiede gibt es zwischen Regression und Klassifikation?",
    "Wie funktioniert Clustering?",
    "Was ist ein Embedding im ML-Kontext?",
    "Welche Bedeutung hat Feature Selection?",
    "Wie kann man die Performanz von ML-Modellen steigern?",
    "Was ist Reinforcement Learning und wofür wird es genutzt?",
    "Wie kann man Daten für das Training augmentieren?",
    "Was ist eine ROC-Kurve?",
    "Wie funktioniert k-Nearest Neighbors?",
    "Wie werden Modelle mit Transfer Learning weiterentwickelt?",
    "Was sind GANs (Generative Adversarial Networks)?",
    "Wie lässt sich die Modellgüte bewerten?",
    "Welche Bedeutung haben Labels im Machine Learning?",
    "Wie funktioniert Cross-Validation?",
    "Was ist eine Epoch im Training eines ML-Modells?",
    "Wie kann man Modelle interpretierbar machen?",
    "Was sind die wichtigsten Anwendungsfälle für ML in der Industrie?",
    "Wie kann man Daten für maschinelles Lernen anonymisieren?",
    "Was versteht man unter Batch Learning?",
    "Was ist Semi-Supervised Learning?",
    "Wie können Ausreißer im Datensatz erkannt und behandelt werden?",
    "Was sind Herausforderungen bei Multi-Label-Klassifikation?",
    "Nenne 3 Vorteile von KI in der Industrie.",
    "Gib mir 5 Beispiele für Machine-Learning-Algorithmen.",
    "Zähle 10 Anwendungen für Deep Learning auf.",
    "Was sind die 2 Hauptarten von neuronalen Netzen?",
    "Welche 4 Schritte umfasst der Data-Science-Prozess?",
    "Gib 6 Methoden zur Datenvorverarbeitung an.",
    "Was sind 3 Herausforderungen bei der Datensicherheit?",
    "Nenne 7 bekannte Frameworks für KI.",
    "Welche 8 Programmiersprachen werden im ML eingesetzt?",
    "Zähle 5 Nachteile von Black-Box-Modellen auf.",
    "Erkläre in 2 Sätzen den Begriff Overfitting.",
    "Gib 10 Tipps zur Optimierung von Trainingsdaten.",
    "Was sind 4 Kennzahlen zur Bewertung von ML-Modellen?",
    "Nenne 5 Ursachen für Bias in Datensätzen.",
    "Was sind die 3 beliebtesten Cloud-Anbieter für AIaaS?",
    "Beschreibe 6 typische Anwendungsbereiche von NLP.",
    "Welche 2 Techniken zur Regularisierung gibt es?",
    "Gib mir 9 Vorteile von Transfer Learning.",
    "Was sind 5 ethische Aspekte beim Einsatz von KI?",
    "Nenne 8 Herausforderungen beim Einsatz von KI in KMUs."
]

def enrich_prompt(prompt, ausfuehrlich=True):
    """Fügt ein zufälliges Stil-Schlüsselwort in den Prompt ein, an sinnvoller Stelle."""
    if ausfuehrlich:
        kw = random.choice(keywords_ausfuehrlich)
        als_befehl = ["erkläre", "beschreibe", "nennen", "erläutere", "erörtere"]
        if kw in als_befehl:
            if prompt.lower().startswith(("was", "wie", "wo", "warum")):
                return f"{kw.capitalize()}, {prompt[0].lower() + prompt[1:]}"
            else:
                return f"{kw.capitalize()} {prompt}"
        else:
            return f"{prompt} ({kw})"
    else:
        kw = random.choice(keywords_kurz)
        return f"{prompt} ({kw})"

output_file = "chatbot_energieverbrauch_v3.csv"
header_needed = not os.path.exists(output_file)

for idx, prompt in enumerate(base_prompts):
    for ausfuehrlich, modellname in zip([True, False], ["Chatbot 1 lang", "Chatbot 2 kurz"]):
        new_prompt = enrich_prompt(prompt, ausfuehrlich=ausfuehrlich)
        tracker = EmissionsTracker(project_name="ollama-chatbot")
        tracker.start()
        start_time = time.time()
        try:
            antwort = generate_ollama_response(new_prompt)
        except Exception as e:
            print(f"Fehler bei Prompt: {new_prompt}, Stil: {modellname} - {e}")
            tracker.stop()
            continue
        emissions = tracker.stop()
        dauer = time.time() - start_time

        prompt_len = len(new_prompt)
        prompt_words = count_words(new_prompt)
        keyword_ausfuehrlich = has_keyword(new_prompt, keywords_ausfuehrlich)
        keyword_kurz = has_keyword(new_prompt, keywords_kurz)
        num_special_chars = count_special_chars(new_prompt)
        num_numbers = count_numbers(new_prompt)

        if keyword_ausfuehrlich and keyword_kurz:
            print(f"WARNUNG: Beide Keywords erkannt im Prompt: {new_prompt}")

        result = {
            "Modell": modellname,
            "Prompt": new_prompt,
            "Prompt_Laenge": prompt_len,
            "Prompt_Woerter": prompt_words,
            "Enthaelt_Ausfuehrlich_Schluesselwort": keyword_ausfuehrlich,
            "Enthaelt_Kurz_Schluesselwort": keyword_kurz,
            "Anzahl_Sonderzeichen": num_special_chars,
            "Anzahl_Zahlen": num_numbers,
            "Energieverbrauch": emissions,
            "Dauer (Sekunden)": dauer
        }

        df = pd.DataFrame([result])
        df.to_csv(output_file, mode='a', header=header_needed, index=False, encoding='utf-8')
        header_needed = False

        print(f"{modellname} - Prompt {idx + 1}/{len(base_prompts)} fertig.")
        time.sleep(1)

print("CSV gespeichert!")
