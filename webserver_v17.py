from flask import Flask, render_template, request, Response, jsonify
import requests
import torch
from torchvision import models, transforms
from PIL import Image
import io, json, base64
import numpy as np
from captum.attr import Saliency, NoiseTunnel
from codecarbon import EmissionsTracker
import os
import faiss
import matplotlib.pyplot as plt
import PyPDF2
import joblib
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE = Image.BILINEAR

# Sprachmodell
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DAMAGED_CLASSES = ['damaged', 'intact']

# BERT-Modell laden
ENERGY_MODEL = joblib.load("rf_energy_embed.joblib")
bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Prompt-Energie-Vorhersage
def predict_emission(prompt):
    embedding = bert_model.encode([prompt])
    return float(ENERGY_MODEL.predict(embedding)[0])

# Modellauswahl
MODEL_CONFIGS = {
    'densenet': (models.densenet121, 'damaged_packages_densenet_epoch.pth'),
    'efficientnet': (models.efficientnet_b0, 'damaged_packages_efficientnet_epoch.pth'),
    'mobilenetv3': (models.mobilenet_v3_large, 'damaged_packages_mobilenetv3_epoch.pth'),
    'resnet18': (models.resnet18, 'damaged_packages_resnet18_epoch.pth'),
    'resnet50': (models.resnet50, 'damaged_packages_resnet50_epoch.pth'),
}

ALL_MODELS = {}
FAILED_MODELS = []

for name, (init_fn, weight_file) in MODEL_CONFIGS.items():
    try:
        model = init_fn(weights=None)
        if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Linear):
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, 2)
        elif hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        model.eval()
        ALL_MODELS[name] = model
    except Exception as e:
        print(f"Fehler beim Laden von {name}: {e}")
        FAILED_MODELS.append(name)

STANDARD_MEAN = [0.485, 0.456, 0.406]
STANDARD_STD = [0.229, 0.224, 0.225]
STANDARD_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=STANDARD_MEAN, std=STANDARD_STD)
])

MODEL_TRANSFORMS = {
    "efficientnet": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=STANDARD_MEAN, std=STANDARD_STD)
    ]),
    "densenet": STANDARD_TRANSFORM,
    "mobilenetv3": STANDARD_TRANSFORM,
    "resnet18": STANDARD_TRANSFORM,
    "resnet50": STANDARD_TRANSFORM,
}

index = None
chunk_store = None
embedding_model = None

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/metrics')
def metrics_page():
    return render_template('metrics.html')

@app.route('/sla')
def sla_page():
    return render_template('sla.html')

@app.route('/privacy')
def privacy_page():
    return render_template('privacy.html')

@app.route('/ethics')
def ethics_page():
    return render_template('ethics.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/documentation')
def documentation_page():
    return render_template('documentation.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global index, chunk_store, embedding_model

    if 'pdf' not in request.files:
        return jsonify(error="Keine PDF-Datei ausgewählt."), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify(error="Keine Datei ausgewählt."), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify(error="Nur PDF-Dateien erlaubt."), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            all_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n"
    except Exception as e:
        return jsonify(error=f"Fehler beim Auslesen der PDF: {e}"), 500

    chunk_size = 500
    overlap = 50
    text_chunks = []
    for i in range(0, len(all_text), chunk_size - overlap):
        chunk = all_text[i:i + chunk_size].strip()
        if chunk:
            text_chunks.append(chunk)

    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    new_embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    pdf_chunks = [f"{c}\n[PDF: {file.filename}]" for c in text_chunks]

    if index is None:
        index = faiss.IndexFlatL2(new_embeddings.shape[1])
    if chunk_store is None:
        chunk_store = []

    index.add(new_embeddings)
    chunk_store.extend(pdf_chunks)

    return jsonify(success=True, num_chunks=len(text_chunks))

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
    global index, chunk_store, embedding_model

    data = request.get_json()
    msg = data.get('message')
    if not msg:
        return jsonify(error="Keine Nachricht angegeben."), 400

    # ML-Vorhersage des Energieverbrauchs
    predicted_emissions = predict_emission(msg)

    task_tracker = EmissionsTracker(project_name="chat_request", measure_power_secs=5)
    task_tracker.start()

    def generate():
        try:
            context = ""
            if (
                    index is not None
                    and chunk_store is not None
                    and embedding_model is not None
                    and len(chunk_store) > 0
            ):
                query_embedding = embedding_model.encode([msg], convert_to_numpy=True)
                D, I = index.search(query_embedding, k=3)
                context = "\n".join([chunk_store[i] for i in I[0] if D[0][I[0].tolist().index(i)] < 1.0])

            if context.strip():
                prompt = (
                    "Du bist ein hilfreicher, deutschsprachiger KI-Chatassistent. "
                    "Nutze für deine Antwort nach Möglichkeit die folgenden Kontextinformationen (z.B. aus PDFs), wenn sie relevant für die Frage sind. "
                    "Ergänze die Antwort ggf. mit deinem eigenen Wissen, falls der Kontext keine vollständige Antwort enthält. "
                    "Wenn du Kontext verwendest, gib für jede Information bitte an, aus welcher Quelle sie stammt – zum Beispiel [PDF: Dateiname]. "
                    "Falls im bereitgestellten Kontext keine relevanten Informationen enthalten sind, antworte wie gewohnt und gib dein eigenes Wissen wieder.\n\n"
                    f"Kontext:\n{context}\n"
                    f"Frage:\n{msg}\n"
                    "Antwort:"
                )
            else:
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
                        emissions = task_tracker.stop()
                        yield f"data: {json.dumps({'emissions': emissions, 'predicted_emissions': predicted_emissions})}\n\n"
                        yield "data: [DONE]\n\n"
                        emissions_reported = True
                        break
            if not emissions_reported:
                emissions = task_tracker.stop()
                yield f"data: {json.dumps({'emissions': emissions, 'predicted_emissions': predicted_emissions})}\n\n"
        except requests.Timeout:
            emissions = task_tracker.stop()
            yield f"data: {json.dumps({'error': 'Anfrage an Ollama zu lange.', 'emissions': emissions, 'predicted_emissions': predicted_emissions})}\n\n"
        except Exception as e:
            emissions = task_tracker.stop()
            yield f"data: {json.dumps({'error': str(e), 'emissions': emissions, 'predicted_emissions': predicted_emissions})}\n\n"

    return Response(generate(), content_type='text/event-stream')

def letterbox(im, new_size=224, color=(0, 0, 0)):
    w, h = im.size
    scale = new_size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    im_resized = im.resize((nw, nh), resample=RESAMPLE)
    new_im = Image.new('RGB', (new_size, new_size), color)
    new_im.paste(im_resized, ((new_size - nw) // 2, (new_size - nh) // 2))
    return new_im

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify(error="Kein Bild ausgewählt."), 400

    model_name = request.form.get('model', 'efficientnet').lower()
    if model_name not in ALL_MODELS:
        return jsonify(error=f"Ungültiger Modellname: {model_name}"), 400
    model = ALL_MODELS[model_name]
    transform = MODEL_TRANSFORMS[model_name]

    task_tracker = EmissionsTracker(project_name="img_classification", measure_power_secs=5)
    task_tracker.start()

    img_bytes = request.files['image'].read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        proc_img = letterbox(image, new_size=224)
        W, H = proc_img.size

        inp = transform(proc_img).unsqueeze(0)
        inp.requires_grad_(True)

        out = model(inp)
        probs = torch.nn.functional.softmax(out, dim=1)
        prob = torch.max(probs, 1)
        idx = prob.indices.item()
        cls = DAMAGED_CLASSES[idx]

        sal = Saliency(model)
        nt = NoiseTunnel(sal)
        attr = nt.attribute(inp, nt_type='smoothgrad', stdevs=0.1, nt_samples=35, target=idx)

        arr = attr.squeeze().cpu().detach().numpy()
        arr = np.abs(arr).max(axis=0)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

        cmap = plt.get_cmap('jet')
        heatmap = cmap(arr)
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        heat_img = Image.fromarray(heatmap).resize((W, H), resample=RESAMPLE).convert("RGBA")
        heat_img.putalpha(255)
        base = proc_img.convert("RGBA")

        def to_url(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        emissions = task_tracker.stop()

        return jsonify({
            "classification": cls,
            "confidence": round(prob.values.item() * 100, 2),
            "original_image": to_url(base),
            "saliency_overlay": to_url(heat_img),
            "emissions": emissions
        })
    except Exception as e:
        emissions = task_tracker.stop()
        return jsonify(error=f"Fehler Bildverarbeitung: {e}", emissions=emissions), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
