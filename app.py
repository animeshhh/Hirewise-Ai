import os
import sys
import json
import re
from flask import Flask, request, render_template
from pypdf import PdfReader
from resumeparser import ats_extractor
from predictor import extract_features, predict_performance

sys.path.insert(0, os.path.abspath(os.getcwd()))
UPLOAD_PATH = r"__DATA__"
os.makedirs(UPLOAD_PATH, exist_ok=True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', data=None)

@app.route("/process", methods=["POST"])
def ats():
    doc = request.files['pdf_doc']
    doc.save(os.path.join(UPLOAD_PATH, "file.pdf"))
    doc_path = os.path.join(UPLOAD_PATH, "file.pdf")

    data = _read_file_from_path(doc_path)
    result = ats_extractor(data)

    try:
        cleaned = re.search(r"\{.*\}", result, re.DOTALL).group(0)
        json_data = json.loads(cleaned)
    except Exception as e:
        json_data = {
            "error": f"Could not parse response: {str(e)}",
            "raw_response": result
        }
        return render_template('index.html', data=json_data)

    # Debugging: Print the parsed resume data to ensure it's correct
    print(f"Parsed Resume Data: {json_data}")

    # Extract features from the parsed resume data
    features = extract_features(json_data)

    # Debugging: Print the extracted features to verify they vary between resumes
    print(f"Extracted Features: {features}")

    # Predict performance using the extracted features
    prediction = predict_performance(features)

    # Adding prediction results to the JSON response
    json_data['prediction_results'] = prediction['scores']
    json_data['best_model'] = prediction['best_model']

    return render_template('index.html', data=json_data)

def _read_file_from_path(path):
    reader = PdfReader(path)
    data = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            data += text
    return data

if __name__ == "__main__":
    app.run(port=8000, debug=True)
