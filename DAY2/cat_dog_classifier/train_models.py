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
from skimage.feature import hog

BASE_DIR = Path(__file__).parent

def extract_features(img):
    """Extract HOG + color histogram features"""
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # HOG features
    hog_feat=hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),feature_vector=True)
    
    # Color histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)
    
    return np.concatenate([hog_feat, hist_h, hist_s])

def load_dataset(num_samples=2000):
    print("Loading cats_vs_dogs dataset...")
    dataset = hf_load_dataset("cats_vs_dogs", split="train").shuffle(seed=42)
    
    X, y = [], []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        if i % 500 == 0:
            print(f"Processing {i}/{num_samples}...")
        img = np.array(item['image'].convert('RGB'))
        X.append(extract_features(img))
        y.append(item['labels'])
    
    return np.array(X), np.array(y)

def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    print("\nTraining SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    models['svm'] = svm
    print(f"SVM Accuracy: {accuracy_score(y_test, svm.predict(X_test_scaled)):.4f}")
    
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    models['random_forest'] = rf
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test_scaled)):.4f}")
    
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['logistic_regression'] = lr
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr.predict(X_test_scaled)):.4f}")
    
    print("\nTraining K-means...")
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
    kmeans.fit(X_train_scaled)
    cluster_labels = kmeans.predict(X_test_scaled)
    cluster_to_label = {}
    for c in range(2):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_to_label[c] = np.bincount(y_test[mask]).argmax()
    models['kmeans'] = kmeans
    models['kmeans_cluster_mapping'] = cluster_to_label
    y_pred = np.array([cluster_to_label.get(c, 0) for c in cluster_labels])
    print(f"K-means Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    models['scaler'] = scaler
    return models

def save_models(models):
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    
    for name, model in models.items():
        if name not in ['scaler', 'kmeans_cluster_mapping']:
            with open(model_dir / f"{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
    
    with open(model_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(models['scaler'], f)
    with open(model_dir / "kmeans_mapping.pkl", 'wb') as f:
        pickle.dump(models['kmeans_cluster_mapping'], f)
    
    print("\nModels saved!")

if __name__ == "__main__":
    print("="*50)
    print("Cat vs Dog Classifier - Training")
    print("="*50)
    
    X, y = load_dataset(num_samples=2000)
    print(f"\nLoaded {len(X)} images, {X.shape[1]} features")
    print(f"Cats: {sum(y==0)}, Dogs: {sum(y==1)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = train_models(X_train, X_test, y_train, y_test)
    save_models(models)
    
    print("\nDone! Run: python app.py")
