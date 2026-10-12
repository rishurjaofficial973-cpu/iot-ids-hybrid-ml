import os
import json
import time
import joblib
import pandas as pd
import numpy as np

class HybridIDSEngine:
    """
    Hybrid Decision Engine combining Supervised Multi-Class Random Forest signature classification
    with Unsupervised Bottleneck Autoencoder reconstruction error scoring for Zero-Day Anomaly Detection.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.rf_model = None
        self.autoencoder = None
        self.scaler = None
        self.label_encoder = None
        self.selected_features = []
        self.threshold = 0.015
        self.ae_type = "sklearn"
        self._load_artifacts()

    def _load_artifacts(self):
        features_json = os.path.join(self.models_dir, "selected_features.json")
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        le_path = os.path.join(self.models_dir, "label_encoder.joblib")
        rf_path = os.path.join(self.models_dir, "rf_model.joblib")
        ae_metadata_path = os.path.join(self.models_dir, "ae_metadata.json")

        if not all(os.path.exists(p) for p in [features_json, scaler_path, le_path, rf_path, ae_metadata_path]):
            raise FileNotFoundError("One or more required model artifacts are missing. Run training pipeline first.")

        with open(features_json, 'r') as f:
            self.selected_features = json.load(f)['selected_features']

        with open(ae_metadata_path, 'r') as f:
            meta = json.load(f)
            self.threshold = meta.get('reconstruction_threshold_p99', 0.015)
            self.ae_type = meta.get('autoencoder_type', 'sklearn')

        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(le_path)
        self.rf_model = joblib.load(rf_path)

        ae_keras_path = os.path.join(self.models_dir, "autoencoder.keras")
        ae_joblib_path = os.path.join(self.models_dir, "autoencoder.joblib")

        if os.path.exists(ae_keras_path):
            try:
                from tensorflow import keras
                self.autoencoder = keras.models.load_model(ae_keras_path)
                self.ae_type = "keras"
            except Exception:
                if os.path.exists(ae_joblib_path):
                    self.autoencoder = joblib.load(ae_joblib_path)
                    self.ae_type = "sklearn"
        elif os.path.exists(ae_joblib_path):
            self.autoencoder = joblib.load(ae_joblib_path)
            self.ae_type = "sklearn"

    def predict_flow(self, flow_input) -> dict:
        """
        Processes a single flow dictionary or DataFrame row.
        Returns hybrid classification badge, zero-day alert flag, MSE reconstruction error, and CPU latency.
        """
        t0 = time.perf_counter()

        if isinstance(flow_input, dict):
            flow_df = pd.DataFrame([flow_input])
        elif isinstance(flow_input, pd.Series):
            flow_df = pd.DataFrame([flow_input.to_dict()])
        elif isinstance(flow_input, pd.DataFrame):
            flow_df = flow_input.iloc[0:1]
        else:
            raise TypeError("flow_input must be a dict, pd.Series, or pd.DataFrame")

        missing_cols = [c for c in self.selected_features if c not in flow_df.columns]
        if missing_cols:
            for c in missing_cols:
                flow_df[c] = 0.0

        X_raw = flow_df[self.selected_features].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw)

        # 1. Unsupervised Autoencoder Reconstruction MSE Error
        if self.ae_type == "keras" and hasattr(self.autoencoder, 'predict'):
            recon = self.autoencoder.predict(X_scaled, verbose=0)
        else:
            recon = self.autoencoder.predict(X_scaled)

        recon_mse = float(np.mean(np.square(X_scaled - recon)))

        # 2. Supervised Classification & Confidence
        rf_probs = self.rf_model.predict_proba(X_scaled)[0]
        top_idx = int(np.argmax(rf_probs))
        top_label = self.label_encoder.classes_[top_idx]
        confidence = float(rf_probs[top_idx])

        probs_dict = {
            cls: float(prob) for cls, prob in zip(self.label_encoder.classes_, rf_probs)
        }

        # 3. Hybrid Fusion Decision Matrix
        is_high_reconstruction_error = recon_mse > self.threshold
        is_ambiguous = confidence < 0.85

        if is_high_reconstruction_error and (is_ambiguous or top_label == 'BENIGN'):
            final_badge = "POTENTIAL ZERO-DAY ANOMALY"
            classification_type = "Zero-Day Anomaly"
            is_zero_day = True
        else:
            final_badge = top_label
            classification_type = "Signature Classification"
            is_zero_day = False

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return {
            "final_badge": final_badge,
            "classification_type": classification_type,
            "supervised_label": top_label,
            "supervised_confidence": confidence,
            "reconstruction_mse": recon_mse,
            "reconstruction_threshold": self.threshold,
            "is_zero_day": is_zero_day,
            "latency_ms": latency_ms,
            "class_probabilities": probs_dict,
            "scaled_features": X_scaled,
            "raw_features": flow_df[self.selected_features].to_dict(orient='records')[0]
        }

if __name__ == "__main__":
    engine = HybridIDSEngine()
    sample = {f: 0.5 for f in engine.selected_features}
    res = engine.predict_flow(sample)
    print("Sample Flow Hybrid Engine Output:")
    print(json.dumps({k: v for k, v in res.items() if k != 'scaled_features'}, indent=2))
