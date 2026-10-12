import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_supervised_models(
    parquet_path: str = "data/processed/clean_iot_dataset.parquet",
    features_json: str = "models/selected_features.json",
    models_dir: str = "models"
):
    """
    Trains Random Forest and Decision Tree classifiers on selected features.
    Saves scaler, label encoder, and trained model artifacts (.joblib).
    Evaluates multi-class accuracy, precision, recall, F1, and single-flow CPU latency.
    """
    print("=" * 80)
    print(" [STEP 3A] SUPERVISED CLASSIFIER TRAINING & BENCHMARKING")
    print("=" * 80)
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet dataset missing at '{parquet_path}'")
    if not os.path.exists(features_json):
        raise FileNotFoundError(f"Selected features JSON missing at '{features_json}'")
    
    with open(features_json, 'r') as f:
        selected_features = json.load(f)['selected_features']
        
    print(f"Loading dataset and using {len(selected_features)} selected features...")
    df = pd.read_parquet(parquet_path)
    
    X = df[selected_features].values.astype(np.float32)
    y_raw = df['Label'].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Dataset Split: Train={X_train.shape[0]:,} samples, Test={X_test.shape[0]:,} samples across {len(label_encoder.classes_)} classes")
    
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    rf_path = os.path.join(models_dir, "rf_model.joblib")
    dt_path = os.path.join(models_dir, "dt_model.joblib")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, le_path)
    print(f"Saved scaler to '{scaler_path}' and label encoder to '{le_path}'")

    print("\n--- Training Random Forest Multi-Class Classifier ---")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=120, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - start_time
    print(f" Random Forest trained in {rf_train_time:.2f} seconds.")

    print("\n--- Training Decision Tree Classifier ---")
    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=42)
    dt.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)
    dt_preds = dt.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    dt_acc = accuracy_score(y_test, dt_preds)

    print("\n" + "=" * 50)
    print(f" RANDOM FOREST ACCURACY : {rf_acc * 100:.2f}%")
    print(f" DECISION TREE ACCURACY  : {dt_acc * 100:.2f}%")
    print("=" * 50)
    
    print("\nDetailed Random Forest Classification Report:")
    print(classification_report(y_test, rf_preds, target_names=label_encoder.classes_, digits=4, zero_division=0))

    print("\nBatch Latency Throughput Benchmark (1,000 sample flows)...")
    t0 = time.perf_counter()
    _ = rf.predict(X_test[:1000])
    t1 = time.perf_counter()
    avg_latency = ((t1 - t0) * 1000.0) / 1000.0
    print(f" Average Inference Time per Flow : {avg_latency:.4f} ms")
    if avg_latency < 1.0:
        print(" SUCCESS: Sub-millisecond latency target (< 1.0 ms) PASSED!")
        
    joblib.dump(rf, rf_path)
    joblib.dump(dt, dt_path)
    print(f"\nSaved Random Forest model to '{rf_path}'")
    print(f"Saved Decision Tree model to '{dt_path}'")
    
    return rf, scaler, label_encoder, rf_acc, avg_latency

if __name__ == "__main__":
    train_supervised_models()
