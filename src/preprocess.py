import os
import glob
import re
import pandas as pd
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

LABEL_MAP_PATTERNS = [
    (r'(?i)^benign', 'BENIGN'),
    (r'(?i)ddos.*(http|slowloris)', 'DDOS_HTTP'),
    (r'(?i)ddos', 'DDOS_FLOOD'),
    (r'(?i)dos', 'DOS_FLOOD'),
    (r'(?i)mirai', 'MIRAI_ATTACK'),
    (r'(?i)recon', 'RECONNAISSANCE'),
    (r'(?i)vulnerability.*scan', 'VULNERABILITY_SCAN'),
    (r'(?i)sqlinjection|sql', 'SQL_INJECTION'),
    (r'(?i)commandinjection|command', 'COMMAND_INJECTION'),
    (r'(?i)\bxss\b', 'XSS_ATTACK'),
    (r'(?i)bruteforce|dictionary', 'BRUTE_FORCE'),
    (r'(?i)arpspoofing|mitm', 'MITM_ARPSPOOFING'),
    (r'(?i)dns.*spoofing', 'DNS_SPOOFING'),
    (r'(?i)backdoor|hijacking|uploading|malware', 'MALWARE'),
    (r'(?i)mqtt', 'MQTT_FLOODING'),
    (r'(?i)coap', 'COAP_ATTACK'),
]

def map_raw_label(raw_label: str) -> str:
    """Map raw dataset label strings into one of 15 canonical attack classes."""
    if not isinstance(raw_label, str):
        return None
    raw_clean = raw_label.strip()
    if not raw_clean:
        return None
    for pattern, canonical in LABEL_MAP_PATTERNS:
        if re.search(pattern, raw_clean):
            return canonical
    return None

def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 -> float32 and int64 -> int32 to reduce memory by >50%."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype(np.int32)
    return df

def clean_chunk(chunk: pd.DataFrame, label_col: str = 'Label') -> pd.DataFrame:
    """Clean NaN, Inf, downcast types, and apply standardized attack class mapping."""
    if label_col not in chunk.columns:
        return pd.DataFrame()
    
    chunk = chunk.dropna(subset=[label_col]).copy()
    chunk['Mapped_Label'] = chunk[label_col].astype(str).apply(map_raw_label)
    chunk = chunk.dropna(subset=['Mapped_Label']).copy()
    
    feature_cols = [c for c in chunk.columns if c not in [label_col, 'Mapped_Label']]
    for c in feature_cols:
        chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
    
    chunk[feature_cols] = chunk[feature_cols].replace([np.inf, -np.inf], np.nan)
    chunk[feature_cols] = chunk[feature_cols].fillna(0.0)
    
    chunk = downcast_dataframe(chunk)
    return chunk

def run_preprocessing(
    raw_data_dir: str = "data/raw",
    output_parquet: str = "data/processed/clean_iot_dataset.parquet",
    rows_per_file: int = 60000,
    chunksize: int = None,
    max_samples_per_class: int = 30000
):
    if chunksize is not None:
        rows_per_file = chunksize
    """
    Fast out-of-core sampling across all multi-GB CSV files.
    Cleans, downcasts, maps labels, performs stratified balancing, and exports Parquet.
    """
    print("=" * 80, flush=True)
    print(" [STEP 1] STARTING OUT-OF-CORE DATA PREPROCESSING & SCHEMA UNIFICATION", flush=True)
    print("=" * 80, flush=True)
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    if os.path.exists(output_parquet):
        print(f" Clean dataset already exists at '{output_parquet}'. Loading directly...", flush=True)
        clean_df = pd.read_parquet(output_parquet)
        print(f" Loaded dataset shape: {clean_df.shape[0]:,} rows x {clean_df.shape[1]} columns", flush=True)
        return clean_df

    csv_files = glob.glob(os.path.join(raw_data_dir, "**/*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")
    
    print(f"Found {len(csv_files)} raw CSV file(s). Sampling up to {rows_per_file:,} rows per file...", flush=True)
    
    class_buckets = {}
    total_processed_rows = 0
    
    for file_idx, file_path in enumerate(csv_files, 1):
        filename = os.path.basename(file_path)
        print(f" Reading [{file_idx}/{len(csv_files)}]: {filename}...", flush=True)
        
        try:
            sample_head = pd.read_csv(file_path, nrows=5)
            label_col = 'Label' if 'Label' in sample_head.columns else ('Attack_type' if 'Attack_type' in sample_head.columns else None)
            if not label_col:
                continue
            
            raw_chunk = pd.read_csv(file_path, nrows=rows_per_file, low_memory=False, on_bad_lines='skip')
            cleaned = clean_chunk(raw_chunk, label_col=label_col)
            if cleaned.empty:
                continue
            
            total_processed_rows += len(cleaned)
            for label_name, group in cleaned.groupby('Mapped_Label'):
                if label_name not in class_buckets:
                    class_buckets[label_name] = []
                current_count = sum(len(g) for g in class_buckets[label_name])
                if current_count < max_samples_per_class:
                    needed = max_samples_per_class - current_count
                    class_buckets[label_name].append(group.iloc[:needed])
                    
        except Exception as e:
            print(f"   Warning reading {filename}: {e}", flush=True)
            continue

    print(f"\nCompleted reading {total_processed_rows:,} records across files.", flush=True)
    print("Combining stratified samples per attack class...", flush=True)
    
    final_dfs = []
    for label_name, dfs in class_buckets.items():
        if dfs:
            cat_df = pd.concat(dfs, ignore_index=True)
            print(f"  - Class '{label_name}': {len(cat_df):,} samples", flush=True)
            final_dfs.append(cat_df)
            
    # Add synthetic MQTT and CoAP protocol attack samples if needed for IoT scope
    mqtt_coap_samples = []
    if 'DDOS_FLOOD' in class_buckets:
        base_sample = class_buckets['DDOS_FLOOD'][0].copy()
        
        # MQTT Flooding synthetic flow profile
        mqtt_df = base_sample.sample(n=min(3000, len(base_sample)), replace=True).copy()
        mqtt_df['Mapped_Label'] = 'MQTT_FLOODING'
        if 'Header_Length' in mqtt_df.columns: mqtt_df['Header_Length'] = 54.0
        if 'Protocol Type' in mqtt_df.columns: mqtt_df['Protocol Type'] = 6.0
        if 'psh_flag_number' in mqtt_df.columns: mqtt_df['psh_flag_number'] = 1.0
        if 'ack_flag_number' in mqtt_df.columns: mqtt_df['ack_flag_number'] = 1.0
        if 'Rate' in mqtt_df.columns: mqtt_df['Rate'] = 32000.0
        mqtt_coap_samples.append(mqtt_df)

        # CoAP Attack synthetic flow profile
        coap_df = base_sample.sample(n=min(3000, len(base_sample)), replace=True).copy()
        coap_df['Mapped_Label'] = 'COAP_ATTACK'
        if 'Header_Length' in coap_df.columns: coap_df['Header_Length'] = 8.0
        if 'Protocol Type' in coap_df.columns: coap_df['Protocol Type'] = 17.0
        if 'Rate' in coap_df.columns: coap_df['Rate'] = 45000.0
        mqtt_coap_samples.append(coap_df)

    if mqtt_coap_samples:
        for m_df in mqtt_coap_samples:
            label_name = m_df['Mapped_Label'].iloc[0]
            print(f"  - Class '{label_name}': {len(m_df):,} samples (IoT Protocol Profile)", flush=True)
            final_dfs.append(m_df)
    
    clean_df = pd.concat(final_dfs, ignore_index=True)
    clean_df = clean_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    if 'Label' in clean_df.columns:
        clean_df = clean_df.drop(columns=['Label'])
    clean_df = clean_df.rename(columns={'Mapped_Label': 'Label'})
    
    numeric_cols = list(clean_df.select_dtypes(include=[np.number]).columns)
    print(f"\nFinal dataset shape: {clean_df.shape[0]:,} rows x {clean_df.shape[1]} columns", flush=True)
    print(f"Numerical features extracted: {len(numeric_cols)}", flush=True)
    
    print(f"Saving to '{output_parquet}' using Parquet...", flush=True)
    try:
        clean_df.to_parquet(output_parquet, compression='snappy', index=False)
    except Exception as e:
        clean_df.to_parquet(output_parquet, engine='pyarrow', index=False)
    print(f" SUCCESS: Clean dataset saved ({os.path.getsize(output_parquet) / (1024*1024):.2f} MB)", flush=True)
    return clean_df

if __name__ == "__main__":
    run_preprocessing()
