# 🛡️ Machine Learning-Based Intrusion Detection System for IoT Networks

**Capstone Project ID:** `BT 63` | **Academic Session:** `2026 – 2027`  
**Department of Computer Science & Engineering**, **IILM University, Greater Noida**  
**Under the Guidance of:** Dr. Kamal Kant Verma  

**Team Members:**
- **Rishu Raj** (CS-2341354)
- **Nikita Patle** (CS-23411560)
- **Aliya Sarvar** (CS-2341781)
- **Prateek Singh** (CS-2341315)

---

## 📌 Abstract
The rapid proliferation of Internet of Things (IoT) devices has expanded the cyber-attack surface, exposing networks to intrusions such as Distributed Denial-of-Service (DDoS), spoofing, and injection attacks. Existing Intrusion Detection Systems (IDS) face persistent research gaps, including high computational latency, susceptibility to zero-day evasion, degraded accuracy on encrypted traffic, and limited model explainability.

This project implements a **Lightweight Hybrid Machine Learning-based IDS** combining:
1. **Supervised Multi-Class Classifier (Random Forest & Decision Tree)** with dual-metric feature selection for rapid, high-accuracy signature classification.
2. **Unsupervised Deep Bottleneck Autoencoder** trained exclusively on benign traffic for zero-day anomaly discovery via 99th-percentile reconstruction error thresholding.
3. **Real-Time SHAP (SHapley Additive exPlanations)** local feature attribution (<10 ms latency) for transparent, human-interpretable security alerts.
4. **Interactive Security Operations Center (SOC) Dashboard** built using Streamlit for real-time traffic simulation, inspection, and CSV audit logging.

Evaluated on the benchmark **Edge-IIoTset** dataset across 16 canonical attack vectors, achieving sub-millisecond inference (<0.1 ms per flow) with zero proprietary hardware requirements.

---

## 🏛️ System Architecture

```
[Raw Network Traffic / Flow]
           │
           ▼
[STEP 1: Out-of-Core Preprocessing] ──► Downcasting (float32/int32), Stratified Sampling (205k rows)
           │
           ▼
[STEP 2: Dual Feature Selection]    ──► 60% RF Gini + 40% Mutual Information (40 -> 25 Features)
           │
     ┌─────┴─────────────────────────────────┐
     ▼                                       ▼
[Branch A: Supervised Signature]      [Branch B: Unsupervised Autoencoder]
Random Forest Classifier (120 Trees)  Bottleneck AE (25 -> 12 -> 25)
Inference: < 0.05 ms / flow            MSE Reconstruction Error vs p99 Threshold (0.0622)
     │                                       │
     └─────────────────┬─────────────────────┘
                       │
                       ▼
          [Hybrid Decision Fusion Engine]
          ├── MSE > Threshold & Ambiguous -> 🟠 POTENTIAL ZERO-DAY ANOMALY
          └── Signature Match -> 🔴 ATTACK / 🟢 BENIGN
                       │
                       ▼
          [Real-Time SHAP Local Attribution]
          TreeExplainer + 100-sample background (<10 ms waterfall plots)
                       │
                       ▼
          [Interactive Streamlit SOC Dashboard]
```

---

## 🚀 Key Performance Benchmarks

| Metric | Target | Achieved Performance | Status |
| :--- | :--- | :--- | :--- |
| **Single-Flow CPU Latency** | $< 1.0\text{ ms}$ | **0.024 – 0.051 ms** |  Passed |
| **Local SHAP Attribution Latency** | $< 10.0\text{ ms}$ | **5.4 – 8.9 ms** |  Passed |
| **Mirai Botnet F1-Score** | $> 95\%$ | **99.8%** |  Passed |
| **MQTT Flooding F1-Score** | $> 95\%$ | **100.0%** |  Passed |
| **CoAP Attack F1-Score** | $> 95\%$ | **100.0%** |  Passed |
| **Autoencoder Benign Reconstruct MSE** | Baseline | **0.0088** (p99 threshold: 0.0622) |  Calibrated |

---

## 📂 Project Repository Structure

```
├── dashboard/
│   └── app.py                # Streamlit SOC interactive web dashboard
├── data/
│   ├── processed/            # Cleaned Parquet dataset (clean_iot_dataset.parquet)
│   └── raw/                  # Edge-IIoTset benchmark raw CSVs (ignored from Git)
├── models/
│   ├── ae_metadata.json      # Autoencoder dimensions & calibrated MSE thresholds
│   └── selected_features.json# Ranked top 25 selected features & weights
├── src/
│   ├── preprocess.py         # Out-of-core cleaning & stratified sampling pipeline
│   ├── feature_selection.py  # Dual feature reduction (Gini + Mutual Information)
│   ├── train_supervised.py   # Random Forest & Decision Tree multi-class training
│   ├── train_autoencoder.py  # Unsupervised bottleneck autoencoder training
│   ├── fusion.py             # Hybrid decision engine combining signature & anomaly
│   └── explainability.py     # Real-time SHAP TreeExplainer & waterfall generator
├── requirements.txt          # Python dependencies
├── run_pipeline.py           # End-to-end automated ML pipeline execution script
└── README.md                 # Complete project documentation
```

---

## 💻 Quick Start & Execution

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/rishurjaofficial973-cpu/Machine-Learning-Based-Intrusion-Detection-System-for-IoT-Networks.git
cd Machine-Learning-Based-Intrusion-Detection-System-for-IoT-Networks
pip install -r requirements.txt
```

### 2. Run the Full ML Training Pipeline
To execute end-to-end data cleaning, feature ranking, model training, and threshold calibration:
```bash
python run_pipeline.py
```

### 3. Launch the Interactive SOC Dashboard
```bash
python -m streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501` to test live presets, inject custom flows, view SHAP attributions, and export audit logs.

---

## 📜 Academic Citation
```bibtex
@capstone{iot_ids_bt63_2026,
  author    = {Rishu Raj and Nikita Patle and Aliya Sarvar and Prateek Singh},
  title     = {Machine Learning-Based Intrusion Detection System for IoT Networks},
  institution = {IILM University, Greater Noida},
  supervisor = {Dr. Kamal Kant Verma},
  year      = {2026-2027},
  note      = {Capstone Project ID: BT 63}
}
```
