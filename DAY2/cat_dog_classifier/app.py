from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pickle
from pathlib import Path
import os
from skimage.feature import hog

BASE_DIR = Path(__file__).parent

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = BASE_DIR / 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

models = {}
scaler = None
kmeans_mapping = None

def load_models():
    global models, scaler, kmeans_mapping
    model_dir = BASE_DIR / "models"
    
    with open(model_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / "svm.pkl", 'rb') as f:
        models['svm'] = pickle.load(f)
    with open(model_dir / "random_forest.pkl", 'rb') as f:
        models['random_forest'] = pickle.load(f)
    with open(model_dir / "logistic_regression.pkl", 'rb') as f:
        models['logistic_regression'] = pickle.load(f)
    with open(model_dir / "kmeans.pkl", 'rb') as f:
        models['kmeans'] = pickle.load(f)
    with open(model_dir / "kmeans_mapping.pkl", 'rb') as f:
        kmeans_mapping = pickle.load(f)

def extract_features(img):
    """Extract HOG + color histogram (must match training)"""
    img = cv2.resize(img, (64, 64))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), feature_vector=True)
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)
    
    return np.concatenate([hog_feat, hist_h, hist_s]).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        model_name = request.form.get('model', 'svm')
        
        filepath = app.config['UPLOAD_FOLDER'] / file.filename
        file.save(str(filepath))
        
        img = cv2.imread(str(filepath))
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        features = extract_features(img)
        features_scaled = scaler.transform(features)
        model = models[model_name]
        
        if model_name == 'kmeans':
            cluster = model.predict(features_scaled)[0]
            prediction = kmeans_mapping.get(cluster, 0)
            probability = None
        else:
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
        
        label = 'Dog' if prediction == 1 else 'Cat'
        os.remove(str(filepath))
        
        return jsonify({
            'prediction': label,
            'model': model_name.replace('_', ' ').title(),
            'confidence': float(max(probability)) if probability is not None else None
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000)