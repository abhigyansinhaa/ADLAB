import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datasets import load_dataset as hf_load_dataset

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent

def load_dataset(num_samples=1000):
    # Load cats_vs_dogs dataset from Hugging Face (shuffled for balanced classes)
    dataset = hf_load_dataset("cats_vs_dogs", split="train").shuffle(seed=42)
    
    X = []
    y = []
    
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        # Get image and preprocess with OpenCV
        img = np.array(item['image'].convert('RGB'))
        img = cv2.resize(img, (64, 64))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_flattened = img_gray.flatten()
        img_normalized = img_flattened / 255.0
        X.append(img_normalized)
        y.append(item['labels'])
    
    return np.array(X), np.array(y)

def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    # SVM
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    models['svm'] = svm
    print(f"SVM Accuracy: {accuracy_score(y_test, svm.predict(X_test_scaled)):.4f}")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    models['random_forest'] = rf
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test_scaled)):.4f}")
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['logistic_regression'] = lr
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr.predict(X_test_scaled)):.4f}")
    
    # K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train_scaled)
    cluster_labels = kmeans.predict(X_test_scaled)
    cluster_to_label = {}
    for cluster_id in range(2):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            cluster_to_label[cluster_id] = np.bincount(y_test[mask]).argmax()
    models['kmeans'] = kmeans
    models['kmeans_cluster_mapping'] = cluster_to_label
    y_pred_kmeans = np.array([cluster_to_label.get(c, 0) for c in cluster_labels])
    print(f"K-means Accuracy: {accuracy_score(y_test, y_pred_kmeans):.4f}")
    
    models['scaler'] = scaler
    return models

def save_models(models, model_dir=None):
    if model_dir is None:
        model_dir = BASE_DIR / "models"
    else:
        model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    for model_name, model in models.items():
        if model_name not in ['scaler', 'kmeans_cluster_mapping']:
            with open(model_dir / f"{model_name}.pkl", 'wb') as f:
                pickle.dump(model, f)
    
    with open(model_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(models['scaler'], f)
    
    if 'kmeans_cluster_mapping' in models:
        with open(model_dir / "kmeans_mapping.pkl", 'wb') as f:
            pickle.dump(models['kmeans_cluster_mapping'], f)
    
    print("Models saved successfully!")

if __name__ == "__main__":
    print("Loading dataset from Hugging Face...")
    X, y = load_dataset(num_samples=1000)  # Adjust sample size as needed
    print(f"Loaded {len(X)} images")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training models...")
    models = train_models(X_train, X_test, y_train, y_test)
    
    save_models(models)
    print("Done!")