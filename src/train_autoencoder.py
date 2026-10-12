import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def train_unsupervised_autoencoder(
    parquet_path: str = "data/processed/clean_iot_dataset.parquet",
    features_json: str = "models/selected_features.json",
    scaler_path: str = "models/scaler.joblib",
    models_dir: str = "models"
):
    """
    Trains an Unsupervised Deep Bottleneck Autoencoder trained strictly on BENIGN traffic.
    Calculates 99th percentile MSE reconstruction error threshold for zero-day anomaly detection.
    """
    print("=" * 80)
    print(" [STEP 3B] UNSUPERVISED BENIGN AUTOENCODER & THRESHOLD CALIBRATION")
    print("=" * 80)
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet dataset missing at '{parquet_path}'")
    if not os.path.exists(features_json):
        raise FileNotFoundError(f"Selected features JSON missing at '{features_json}'")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler missing at '{scaler_path}'. Run supervised training first.")
        
    with open(features_json, 'r') as f:
        selected_features = json.load(f)['selected_features']
        
    scaler = joblib.load(scaler_path)
    
    print("Loading dataset and filtering strictly BENIGN traffic...")
    df = pd.read_parquet(parquet_path)
    benign_df = df[df['Label'] == 'BENIGN']
    
    if benign_df.empty:
        raise ValueError("No BENIGN samples found in dataset for autoencoder training.")
        
    print(f"Found {len(benign_df):,} BENIGN flow samples.")
    
    X_benign = benign_df[selected_features].values.astype(np.float32)
    X_benign_scaled = scaler.transform(X_benign)
    
    X_train_benign, X_val_benign = train_test_split(
        X_benign_scaled, test_size=0.20, random_state=42
    )
    
    n_features = len(selected_features)
    hidden_dim = max(4, n_features // 2)
    print(f"Autoencoder Architecture: Input ({n_features}) -> Bottleneck ({hidden_dim}) -> Output ({n_features})")
    
    autoencoder_model = None
    ae_type = "keras"
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        inputs = keras.Input(shape=(n_features,))
        encoded = layers.Dense(hidden_dim, activation='relu')(inputs)
        decoded = layers.Dense(n_features, activation='linear')(encoded)
        
        autoencoder = keras.Model(inputs=inputs, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        print("Training Keras Bottleneck Autoencoder...")
        autoencoder.fit(
            X_train_benign, X_train_benign,
            epochs=15,
            batch_size=128,
            validation_data=(X_val_benign, X_val_benign),
            verbose=1
        )
        
        ae_save_path = os.path.join(models_dir, "autoencoder.keras")
        autoencoder.save(ae_save_path)
        autoencoder_model = autoencoder
        print(f"Saved Keras Autoencoder to '{ae_save_path}'")
        
    except (ImportError, Exception) as e:
        print(f"Notice: Keras/TF autoencoder fallback engaged ({e}). Using optimized Scikit-Learn MLPRegressor...")
        ae_type = "sklearn"
        mlp_ae = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            activation='relu',
            solver='adam',
            max_iter=40,
            random_state=42
        )
        mlp_ae.fit(X_train_benign, X_train_benign)
        ae_save_path = os.path.join(models_dir, "autoencoder.joblib")
        joblib.dump(mlp_ae, ae_save_path)
        autoencoder_model = mlp_ae
        print(f"Saved sklearn Autoencoder to '{ae_save_path}'")
        
    print("\nCalculating Validation BENIGN Reconstruction MSE Errors...")
    if ae_type == "keras":
        val_reconstructions = autoencoder_model.predict(X_val_benign, verbose=0)
    else:
        val_reconstructions = autoencoder_model.predict(X_val_benign)
        
    val_mse = np.mean(np.square(X_val_benign - val_reconstructions), axis=1)
    
    threshold_p99 = float(np.percentile(val_mse, 99))
    threshold_p95 = float(np.percentile(val_mse, 95))
    mean_mse = float(np.mean(val_mse))
    
    print("\n--- AUTOENCODER ANOMALY THRESHOLD METRICS ---")
    print(f" Mean Benign MSE        : {mean_mse:.6f}")
    print(f" 95th Percentile MSE   : {threshold_p95:.6f}")
    print(f" 99th Percentile MSE (Threshold) : {threshold_p99:.6f}")
    
    metadata_path = os.path.join(models_dir, "ae_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            "reconstruction_threshold_p99": threshold_p99,
            "reconstruction_threshold_p95": threshold_p95,
            "mean_benign_mse": mean_mse,
            "num_features": n_features,
            "bottleneck_dim": hidden_dim,
            "autoencoder_type": ae_type
        }, f, indent=2)
        
    print(f" SUCCESS: Autoencoder metadata saved to '{metadata_path}'")
    return autoencoder_model, threshold_p99

if __name__ == "__main__":
    train_unsupervised_autoencoder()
