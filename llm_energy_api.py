from flask import Flask, request, Response, jsonify
import requests
import joblib
from sentence_transformers import SentenceTransformer
from codecarbon import EmissionsTracker
import json

app = Flask(__name__)

# Ollama LLM/Chatbot API
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Energie-Predictor & BERT-Embedding
BERT_MODEL_PATH = "bert_embed_model"
ENERGY_MODEL_PATH = "rf_energy_embed.joblib"
bert_model = SentenceTransformer(BERT_MODEL_PATH)
energy_model = joblib.load(ENERGY_MODEL_PATH)

def predict_emission(prompt):
    embedding = bert_model.encode([prompt])
    return float(energy_model.predict(embedding)[0])

@app.route('/predict_energy', methods=['POST'])
def predict_energy():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt.strip():
        return jsonify({"error": "Prompt fehlt."}), 400
    try:
        predicted = predict_emission(prompt)
        return jsonify({"predicted_emissions": predicted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('message')
    if not msg:
        return jsonify(error="Keine Nachricht angegeben."), 400

    predicted_emissions = predict_emission(msg)

    tracker = EmissionsTracker(project_name="chat_request", measure_power_secs=5)
    tracker.start()

    def generate():
        try:
            prompt = (
                "Du bist ein hilfreicher, deutschsprachiger KI-Chatassistent. "
                "Beantworte die folgende Frage so gut du kannst, basierend auf deinem Wissen.\n\n"
                f"Frage:\n{msg}\n"
                "Antwort:"
            )

            payload = {"model": "gemma3:1b", "prompt": prompt, "stream": True}
            r = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=30)
            r.raise_for_status()
            emissions_reported = False
            for line in r.iter_lines():
                if line:
                    d = json.loads(line.decode())
                    if 'response' in d:
                        yield f"data: {json.dumps({'response': d['response']})}\n\n"
                    if d.get('done'):
                        emissions = tracker.stop()
                        yield f"data: {json.dumps({'emissions': emissions, 'predicted_emissions': predicted_emissions})}\n\n"
                        yield "data: [DONE]\n\n"
                        emissions_reported = True
                        break
            if not emissions_reported:
                emissions = tracker.stop()
                yield f"data: {json.dumps({'emissions': emissions, 'predicted_emissions': predicted_emissions})}\n\n"
        except requests.Timeout:
            emissions = tracker.stop()
            yield f"data: {json.dumps({'error': 'Anfrage an Ollama zu lange.', 'emissions': emissions, 'predicted_emissions': predicted_emissions})}\n\n"
        except Exception as e:
            emissions = tracker.stop()
            yield f"data: {json.dumps({'error': str(e), 'emissions': emissions, 'predicted_emissions': predicted_emissions})}\n\n"

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
