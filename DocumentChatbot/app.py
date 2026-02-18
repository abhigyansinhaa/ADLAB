import os
import json
import requests
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

document_text = ""
current_model = None
OLLAMA_URL = "http://localhost:11434"


def extract_text_from_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_excel(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df.to_string()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            return jsonify({'status': 'online', 'models': models})
    except:
        pass
    return jsonify({'status': 'offline', 'models': []})


@app.route('/api/select-model', methods=['POST'])
def select_model():
    global current_model
    current_model = request.json.get('model')
    return jsonify({'success': True, 'model': current_model})


@app.route('/api/upload', methods=['POST'])
def upload():
    global document_text
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    if not current_model:
        return jsonify({'error': 'Please select a model first'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        ext = filename.lower().split('.')[-1]
        if ext == 'pdf':
            document_text = extract_text_from_pdf(filepath)
        elif ext in ['docx', 'doc']:
            document_text = extract_text_from_docx(filepath)
        elif ext in ['xlsx', 'xls', 'csv']:
            document_text = extract_text_from_excel(filepath)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        return jsonify({
            'success': True,
            'filename': filename,
            'chars': len(document_text)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    if not document_text:
        return jsonify({'error': 'Upload a document first'}), 400
    
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question'}), 400
    
    # Use only first 2000 chars for speed
    doc_context = document_text[:2000]
    
    prompt = f"""Document:
{doc_context}

Question: {question}
Answer briefly:"""

    def stream():
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": current_model, "prompt": prompt},
                stream=True,
                timeout=600
            ) as r:
                for line in r.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            text = data.get('response', '')
                            if text:
                                yield f"data:{text}\n\n"
                            if data.get('done'):
                                yield "data:[DONE]\n\n"
                                break
                        except:
                            continue
        except Exception as e:
            yield f"data:[ERROR]{str(e)}\n\n"
    
    return Response(stream(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })


@app.route('/api/clear', methods=['POST'])
def clear():
    global document_text
    document_text = ""
    return jsonify({'success': True})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Document Chatbot - http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, threaded=True)