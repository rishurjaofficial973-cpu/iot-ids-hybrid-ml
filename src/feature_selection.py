import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

def run_feature_selection(
    parquet_path: str = "data/processed/clean_iot_dataset.parquet",
    output_json: str = "models/selected_features.json",
    top_k: int = 18,
    sample_size: int = 50000
):
    """
    Loads processed dataset, evaluates Tree-based Gini Importance and Mutual Information,
    and selects top 15-20 low-latency numerical features for IoT IDS.
    """
    print("=" * 80)
    print(" [STEP 2] STARTING DUAL FEATURE SELECTION (RF GINI + MUTUAL INFORMATION)")
    print("=" * 80)
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at '{parquet_path}'. Run preprocessing first.")
    
    print(f"Loading dataset from '{parquet_path}'...")
    df = pd.read_parquet(parquet_path)
    
    if 'Label' not in df.columns:
        raise KeyError("'Label' column missing from parquet file.")
    
    X = df.drop(columns=['Label']).select_dtypes(include=[np.number])
    y = df['Label']
    
    feature_names = list(X.columns)
    print(f"Total candidate numerical features: {len(feature_names)}")
    
    if len(df) > sample_size:
        print(f"Subsampling {sample_size:,} records for fast, memory-friendly feature evaluation...")
        sample_df = df.sample(n=sample_size, random_state=42)
        X_sub = sample_df.drop(columns=['Label']).select_dtypes(include=[np.number])
        y_sub = sample_df['Label']
    else:
        X_sub, y_sub = X, y

    print("\n1. Calculating Random Forest Feature Importances (Gini Impurity)...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_sub, y_sub)
    rf_importances = rf.feature_importances_
    
    print("2. Calculating Mutual Information Scores...")
    mi_scores = mutual_info_classif(X_sub, y_sub, random_state=42)
    
    rf_norm = (rf_importances - rf_importances.min()) / (rf_importances.max() - rf_importances.min() + 1e-8)
    mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
    combined_scores = 0.6 * rf_norm + 0.4 * mi_norm
    
    feature_ranking = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_importances,
        'mi_score': mi_scores,
        'combined_score': combined_scores
    }).sort_values(by='combined_score', ascending=False).reset_index(drop=True)
    
    print("\n--- TOP RANKED FEATURES ---")
    for idx, row in feature_ranking.head(top_k).iterrows():
        print(f" {idx+1:2d}. {row['feature']:<30} | Combined Score: {row['combined_score']:.4f} (RF: {row['rf_importance']:.4f}, MI: {row['mi_score']:.4f})")
    
    selected_features = list(feature_ranking.head(top_k)['feature'])
    
    essential_flags = [f for f in ['syn_flag_number', 'rst_flag_number', 'fin_flag_number', 'psh_flag_number', 'ack_flag_number', 'UDP', 'ICMP', 'HTTP', 'DNS', 'TCP'] if f in feature_names]
    for ef in essential_flags:
        if ef not in selected_features:
            selected_features.append(ef)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump({
            "selected_features": selected_features,
            "num_features": len(selected_features),
            "ranking": feature_ranking.head(top_k).to_dict(orient='records')
        }, f, indent=2)
        
    print(f"\n SUCCESS: Saved top {len(selected_features)} features to '{output_json}'")
    return selected_features

if __name__ == "__main__":
    run_feature_selection()
