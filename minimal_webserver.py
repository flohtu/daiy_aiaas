from flask import Flask, render_template_string, request
import random

app = Flask(__name__)

# random klassifikation
CLASSIFICATION_WORDS = [
    "Hund", "Katze", "Auto", "Baum", "Haus", "Blume", "Vogel", "Sonne", "Mond", "See"
]

# Hauptseite
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DAIY AIaaS Webserver</title>
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      padding: 40px 20px;
      background-color: #f5f6f5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
    }
    .logo {
      float: right;
      max-width: 250px;
      margin-right: 20px;
      margin-bottom: 20px;
    }
    h1 {
      font-size: 2.5rem;
      color: #2d3748;
      margin-bottom: 1rem;
    }
    p {
      color: #4a5568;
      font-size: 1.1rem;
    }
    .service-button {
      width: 100%; 
      padding: 12px;
      font-size: 1rem;
      font-weight: 500;
      margin: 10px 0;
      transition: transform 0.2s, background-color 0.2s;
    }
    .custom-service-button {
      background-color: #84B819;
      color: white;
      border: none;
      border-radius: 8px;
    }
    .custom-service-button:hover {
      background-color: #6E9A14;
      transform: translateY(-2px);
    }
    #imageUploadSection {
      margin-top: 30px;
      background-color: white;
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feedback-buttons {
      margin-top: 15px;
    }
    .alert {
      border-radius: 8px;
    }
    footer {
      background-color: #84B819; 
      color: #ffffff; 
      padding: 20px 0;
      text-align: center;
      margin-top: auto;
    }
    footer a {
      color: #ffffff; 
      text-decoration: none;
      margin: 0 12px;
      font-weight: 500;
    }
    footer a:hover {
      color: #a0d911; 
    }
    footer p {
      margin: 5px 0;
      font-size: 0.9rem;
    }
    @media (max-width: 768px) {
      .logo {
        float: none;
        display: block;
        margin: 0 auto 20px;
      }
      .service-button {
        font-size: 0.9rem;
        padding: 10px;
      }
      h1 {
        font-size: 2rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo" class="logo">
    <h1>DAIY AIaaS Webserver</h1>
    <p>Wähle einen Dienst aus, um fortzufahren:</p>
    <div class="row">
      <div class="col-6 col-md-3">
        <button onclick="showImageUpload()" class="btn custom-service-button service-button">Bildklassifikation</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Textanalyse</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Objekterkennung</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Sprachsynthese</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Predictive Maintenance</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Chatbot</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Gesichtserkennung</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Textgenerierung</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Anomalieerkennung</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Bildsegmentierung</button>
      </div>
      <div class="col-6 col-md-3">
        <button onclick="alert('Dieser Dienst ist noch nicht implementiert.')" class="btn custom-service-button service-button">Empfehlungssystem</button>
      </div>
    </div>

    <!-- Bild-Upload-Bereich -->
    <div id="imageUploadSection" class="d-none">
      <h3>Bildklassifikation</h3>
      <form id="imageUploadForm" enctype="multipart/form-data">
        <input type="file" id="imageInput" name="image" accept="image/*" class="form-control mb-3">
        <button type="button" onclick="uploadImage()" class="btn custom-service-button">Hochladen und Klassifizieren</button>
      </form>
      <div id="result" class="alert alert-info d-none mt-3"></div>
      <div id="feedbackSection" class="feedback-buttons d-none">
        <p>War die Klassifikation korrekt?</p>
        <button onclick="submitFeedback('correct')" class="btn btn-success me-2">Richtig (👍)</button>
        <button onclick="submitFeedback('incorrect')" class="btn btn-danger">Falsch (👎)</button>
      </div>
      <div id="feedbackResult" class="alert alert-info d-none mt-3"></div>
    </div>
  </div>

  <!-- Footer mit SLA und "Trustworthy" AIaaS Links -->
  <footer>
    <p>
      <a href="{{ url_for('sla') }}">Service Level Agreement</a> |
      <a href="{{ url_for('privacy') }}">Datenschutzerklärung</a> |
      <a href="{{ url_for('ethics') }}">Ethik-Richtlinien</a> |
      <a href="{{ url_for('contact') }}">Feedback & Kontakt</a>
    </p>
    <p>© 2025 DAIY AIaaS Webserver. Alle Rechte vorbehalten.</p>
  </footer>

  <!-- Bootstrap JS -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    function showImageUpload() {
      document.getElementById('imageUploadSection').classList.remove('d-none');
      document.getElementById('result').classList.add('d-none');
      document.getElementById('feedbackSection').classList.add('d-none');
      document.getElementById('feedbackResult').classList.add('d-none');
    }

    function uploadImage() {
      const imageInput = document.getElementById('imageInput');
      const resultDiv = document.getElementById('result');
      const feedbackSection = document.getElementById('feedbackSection');

      if (!imageInput.files[0]) {
        resultDiv.textContent = 'Bitte wähle ein Bild aus.';
        resultDiv.classList.remove('d-none');
        feedbackSection.classList.add('d-none');
        return;
      }

      const formData = new FormData();
      formData.append('image', imageInput.files[0]);

      fetch('/classify', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        resultDiv.textContent = `Klassifikationsergebnis: ${data.classification}`;
        resultDiv.classList.remove('d-none');
        feedbackSection.classList.remove('d-none');
      })
      .catch(error => {
        resultDiv.textContent = 'Fehler bei der Klassifikation.';
        resultDiv.classList.remove('d-none');
        feedbackSection.classList.add('d-none');
      });
    }

    function submitFeedback(feedback) {
      const resultDiv = document.getElementById('result').textContent;
      const classification = resultDiv.replace('Klassifikationsergebnis: ', '');
      const feedbackResultDiv = document.getElementById('feedbackResult');

      fetch('/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ classification: classification, feedback: feedback })
      })
      .then(response => response.json())
      .then(data => {
        feedbackResultDiv.textContent = data.message;
        feedbackResultDiv.classList.remove('d-none');
      })
      .catch(error => {
        feedbackResultDiv.textContent = 'Fehler beim Senden des Feedbacks.';
        feedbackResultDiv.classList.remove('d-none');
      });
    }
  </script>
</body>
</html>
"""

# SLA Seite
SLA_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SLA - DAIY AIaaS Webserver</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Roboto', sans-serif; padding: 40px 20px; background-color: #f5f6f5; }
    .container { max-width: 1200px; }
    h1 { color: #2d3748; }
    .custom-service-button { background-color: #84B819; color: white; border: none; border-radius: 8px; }
    .custom-service-button:hover { background-color: #6E9A14; }
    footer {
      background-color: #84B819; 
      color: #ffffff; 
      padding: 20px 0;
      text-align: center;
      margin-top: 40px;
    }
    footer a {
      color: #ffffff; 
      text-decoration: none;
      margin: 0 12px;
      font-weight: 500;
    }
    footer a:hover {
      color: #a0d911; 
    }
    footer p {
      margin: 5px 0;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div class="container mt-5">
    <h1>Service Level Agreement (SLA)</h1>
    <p>Unser SLA garantiert folgende Leistungsstandards:</p>
    <ul>
      <li><strong>Verfügbarkeit</strong>: 99,9% jährliche Verfügbarkeit.</li>
      <li><strong>Reaktionszeit</strong>: Bildklassifikation innerhalb von 10 Sekunden.</li>
      <li><strong>Support</strong>: Antwort auf Feedback innerhalb von 24 Stunden (9:00–17:00 MEZ).</li>
      <li><strong>Wartungsfenster</strong>: Maximal 2 Stunden, 2x jährlich (02:00–04:00 MEZ).</li>
    </ul>
    <p><a href="{{ url_for('index') }}" class="btn custom-service-button">Zurück zur Hauptseite</a></p>
  </div>
  <footer>
    <p>
      <a href="{{ url_for('sla') }}">Service Level Agreement</a> |
      <a href="{{ url_for('privacy') }}">Datenschutzerklärung</a> |
      <a href="{{ url_for('ethics') }}">Ethik-Richtlinien</a> |
      <a href="{{ url_for('contact') }}">Feedback & Kontakt</a>
    </p>
    <p>© 2025 DAIY AIaaS Webserver. Alle Rechte vorbehalten.</p>
  </footer>
</body>
</html>
"""

# Datenschutzerklärung Seite
PRIVACY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Datenschutzerklärung - DAIY AIaaS Webserver</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Roboto', sans-serif; padding: 40px 20px; background-color: #f5f6f5; }
    .container { max-width: 1200px; }
    h1 { color: #2d3748; }
    .custom-service-button { background-color: #84B819; color: white; border: none; border-radius: 8px; }
    .custom-service-button:hover { background-color: #6E9A14; }
    footer {
      background-color: #84B819; 
      color: #ffffff; 
      padding: 20px 0;
      text-align: center;
      margin-top: 40px;
    }
    footer a {
      color: #ffffff; 
      text-decoration: none;
      margin: 0 12px;
      font-weight: 500;
    }
    footer a:hover {
      color: #a0d911; 
    }
    footer p {
      margin: 5px 0;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div class="container mt-5">
    <h1>Datenschutzerklärung</h1>
    <p>Wir nehmen den Schutz deiner Daten ernst:</p>
    <ul>
      <li>Hochgeladene Bilder werden nach der Klassifikation sofort gelöscht.</li>
      <li>Keine Speicherung personenbezogener Daten ohne Zustimmung.</li>
      <li>Alle Datenübertragungen sind verschlüsselt (HTTPS).</li>
      <li>Wir halten uns an die DSGVO und andere Datenschutzgesetze.</li>
    </ul>
    <p><a href="{{ url_for('index') }}" class="btn custom-service-button">Zurück zur Hauptseite</a></p>
  </div>
  <footer>
    <p>
      <a href="{{ url_for('sla') }}">Service Level Agreement</a> |
      <a href="{{ url_for('privacy') }}">Datenschutzerklärung</a> |
      <a href="{{ url_for('ethics') }}">Ethik-Richtlinien</a> |
      <a href="{{ url_for('contact') }}">Feedback & Kontakt</a>
    </p>
    <p>© 2025 DAIY AIaaS Webserver. Alle Rechte vorbehalten.</p>
  </footer>
</body>
</html>
"""

# Ethik-Richtlinien Seite
ETHICS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ethik-Richtlinien - DAIY AIaaS Webserver</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Roboto', sans-serif; padding: 40px 20px; background-color: #f5f6f5; }
    .container { max-width: 1200px; }
    h1 { color: #2d3748; }
    .custom-service-button { background-color: #84B819; color: white; border: none; border-radius: 8px; }
    .custom-service-button:hover { background-color: #6E9A14; }
    footer {
      background-color: #84B819; 
      color: #ffffff; 
      padding: 20px 0;
      text-align: center;
      margin-top: 40px;
    }
    footer a {
      color: #ffffff; 
      text-decoration: none;
      margin: 0 12px;
      font-weight: 500;
    }
    footer a:hover {
      color: #a0d911; 
    }
    footer p {
      margin: 5px 0;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div class="container mt-5">
    <h1>Ethik-Richtlinien</h1>
    <p>Unsere Verpflichtung zu verantwortungsvoller KI:</p>
    <ul>
      <li>Fairness: Minimierung von Bias in unseren Modellen.</li>
      <li>Transparenz: Klare Kommunikation der KI-Entscheidungen.</li>
      <li>Verantwortung: Regelmäßige Audits unserer Systeme.</li>
      <li>Nutzerzentriert: Dein Feedback hilft uns, besser zu werden.</li>
    </ul>
    <p><a href="{{ url_for('index') }}" class="btn custom-service-button">Zurück zur Hauptseite</a></p>
  </div>
  <footer>
    <p>
      <a href="{{ url_for('sla') }}">Service Level Agreement</a> |
      <a href="{{ url_for('privacy') }}">Datenschutzerklärung</a> |
      <a href="{{ url_for('ethics') }}">Ethik-Richtlinien</a> |
      <a href="{{ url_for('contact') }}">Feedback & Kontakt</a>
    </p>
    <p>© 2025 DAIY AIaaS Webserver. Alle Rechte vorbehalten.</p>
  </footer>
</body>
</html>
"""

# Kontakt Seite
CONTACT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Feedback & Kontakt - DAIY AIaaS Webserver</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Roboto', sans-serif; padding: 40px 20px; background-color: #f5f6f5; }
    .container { max-width: 1200px; }
    h1 { color: #2d3748; }
    .custom-service-button { background-color: #84B819; color: white; border: none; border-radius: 8px; }
    .custom-service-button:hover { background-color: #6E9A14; }
    footer {
      background-color: #84B819; 
      color: #ffffff; 
      padding: 20px 0;
      text-align: center;
      margin-top: 40px;
    }
    footer a {
      color: #ffffff; 
      text-decoration: none;
      margin: 0 12px;
      font-weight: 500;
    }
    footer a:hover {
      color: #a0d911; 
    }
    footer p {
      margin: 5px 0;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div class="container mt-5">
    <h1>Feedback & Kontakt</h1>
    <p>Deine Meinung ist uns wichtig. Kontaktiere uns bei Fragen oder Beschwerden:</p>
    <ul>
      <li>E-Mail: <a href="florian.hermann@tu-dortmund.de">florian.hermann@tu-dortmund.de</a></li>
      <li>Telefon: +49 123 456789 (Mo–Fr, 9:00–17:00 MEZ)</li>
    </ul>
    <p><a href="{{ url_for('index') }}" class="btn custom-service-button">Zurück zur Hauptseite</a></p>
  </div>
  <footer>
    <p>
      <a href="{{ url_for('sla') }}">Service Level Agreement</a> |
      <a href="{{ url_for('privacy') }}">Datenschutzerklärung</a> |
      <a href="{{ url_for('ethics') }}">Ethik-Richtlinien</a> |
      <a href="{{ url_for('contact') }}">Feedback & Kontakt</a>
    </p>
    <p>© 2025 DAIY AIaaS Webserver. Alle Rechte vorbehalten.</p>
  </footer>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/sla')
def sla():
    return render_template_string(SLA_HTML)

@app.route('/privacy')
def privacy():
    return render_template_string(PRIVACY_HTML)

@app.route('/ethics')
def ethics():
    return render_template_string(ETHICS_HTML)

@app.route('/contact')
def contact():
    return render_template_string(CONTACT_HTML)

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return {'error': 'Kein Bild hochgeladen'}, 400
    # Simuliere Klassifikation 
    classification = random.choice(CLASSIFICATION_WORDS)
    return {'classification': classification}

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    classification = data.get('classification')
    feedback = data.get('feedback')
    message = f"Vielen Dank für dein Feedback! Du hast '{feedback}' für die Klassifikation '{classification}' gegeben."
    return {'message': message}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
