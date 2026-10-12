import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fusion import HybridIDSEngine
from src.explainability import IDSExplainer

st.set_page_config(
    page_title="IoT IDS - Real-Time Hybrid ML Security Operation Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #888888;
        margin-bottom: 20px;
    }
    .badge-benign {
        background-color: #2e7d32;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    .badge-attack {
        background-color: #c62828;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    .badge-zeroday {
        background-color: #ef6c00;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ids_services():
    engine = HybridIDSEngine(models_dir="models")
    explainer = IDSExplainer(models_dir="models")
    return engine, explainer

try:
    engine, explainer = load_ids_services()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading IDS model artifacts: {e}. Please run 'python run_pipeline.py' first.")

if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

st.markdown('<p class="main-header">🛡️ Lightweight Hybrid ML-IDS for IoT Networks</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time Signature Classification, Zero-Day Anomaly Scoring & SHAP Explainability (MQTT/CoAP)</p>', unsafe_allow_html=True)

st.sidebar.title("🎛️ Flow Simulator & Inputs")
input_mode = st.sidebar.radio("Select Data Source", ["Preset Traffic Scenarios", "Custom Manual Sliders", "Upload PCAP / CSV File"])

features = engine.selected_features if model_loaded else []

preset_dict = {
    "Benign IoT Sensor Flow": {
        'Header_Length': 20.0, 'Protocol Type': 6.0, 'Time_To_Live': 64.0, 'Rate': 45.0,
        'fin_flag_number': 0.0, 'syn_flag_number': 0.0, 'rst_flag_number': 0.0, 'psh_flag_number': 0.0,
        'ack_flag_number': 1.0, 'ece_flag_number': 0.0, 'cwr_flag_number': 0.0, 'ack_count': 5.0,
        'syn_count': 0.0, 'fin_count': 0.0, 'rst_count': 0.0, 'Tot sum': 120.0, 'Min': 40.0,
        'Max': 80.0, 'AVG': 60.0, 'Std': 5.0, 'Tot size': 120.0, 'IAT': 0.05, 'Number': 2.0, 'Variance': 25.0
    },
    "MQTT Publish Flood Attack": {
        'Header_Length': 54.0, 'Protocol Type': 6.0, 'Time_To_Live': 128.0, 'Rate': 35000.0,
        'fin_flag_number': 0.0, 'syn_flag_number': 0.0, 'rst_flag_number': 0.0, 'psh_flag_number': 1.0,
        'ack_flag_number': 1.0, 'ece_flag_number': 0.0, 'cwr_flag_number': 0.0, 'ack_count': 99.0,
        'syn_count': 0.0, 'fin_count': 0.0, 'rst_count': 0.0, 'Tot sum': 8500.0, 'Min': 60.0,
        'Max': 1400.0, 'AVG': 450.0, 'Std': 220.0, 'Tot size': 8500.0, 'IAT': 0.0001, 'Number': 100.0, 'Variance': 48400.0
    },
    "CoAP UDP Flood Attack": {
        'Header_Length': 8.0, 'Protocol Type': 17.0, 'Time_To_Live': 64.0, 'Rate': 48000.0,
        'fin_flag_number': 0.0, 'syn_flag_number': 0.0, 'rst_flag_number': 0.0, 'psh_flag_number': 0.0,
        'ack_flag_number': 0.0, 'ece_flag_number': 0.0, 'cwr_flag_number': 0.0, 'ack_count': 0.0,
        'syn_count': 0.0, 'fin_count': 0.0, 'rst_count': 0.0, 'Tot sum': 15000.0, 'Min': 100.0,
        'Max': 500.0, 'AVG': 300.0, 'Std': 80.0, 'Tot size': 15000.0, 'IAT': 0.00005, 'Number': 100.0, 'Variance': 6400.0
    },
    "DDoS SYN Flood Attack": {
        'Header_Length': 40.0, 'Protocol Type': 6.0, 'Time_To_Live': 255.0, 'Rate': 52000.0,
        'fin_flag_number': 0.0, 'syn_flag_number': 1.0, 'rst_flag_number': 0.0, 'psh_flag_number': 0.0,
        'ack_flag_number': 0.0, 'ece_flag_number': 0.0, 'cwr_flag_number': 0.0, 'ack_count': 0.0,
        'syn_count': 100.0, 'fin_count': 0.0, 'rst_count': 0.0, 'Tot sum': 4000.0, 'Min': 40.0,
        'Max': 40.0, 'AVG': 40.0, 'Std': 0.0, 'Tot size': 4000.0, 'IAT': 0.00002, 'Number': 100.0, 'Variance': 0.0
    },
    "Zero-Day Unseen Anomaly": {
        'Header_Length': 999.0, 'Protocol Type': 254.0, 'Time_To_Live': 1.0, 'Rate': 99999.0,
        'fin_flag_number': 1.0, 'syn_flag_number': 1.0, 'rst_flag_number': 1.0, 'psh_flag_number': 1.0,
        'ack_flag_number': 1.0, 'ece_flag_number': 1.0, 'cwr_flag_number': 1.0, 'ack_count': 999.0,
        'syn_count': 999.0, 'fin_count': 999.0, 'rst_count': 999.0, 'Tot sum': 99999.0, 'Min': 999.0,
        'Max': 9999.0, 'AVG': 5000.0, 'Std': 2500.0, 'Tot size': 99999.0, 'IAT': 0.9999, 'Number': 999.0, 'Variance': 999999.0
    }
}

sample_input = {}

if model_loaded:
    if input_mode == "Preset Traffic Scenarios":
        scenario_choice = st.sidebar.selectbox("Select Scenario Preset", list(preset_dict.keys()))
        sample_input = preset_dict[scenario_choice]
    elif input_mode == "Custom Manual Sliders":
        st.sidebar.markdown("### Feature Customization")
        for f in features:
            sample_input[f] = float(st.sidebar.number_input(f, value=10.0))
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV / Parquet Flow File", type=["csv", "parquet"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_parquet(uploaded_file)
            st.sidebar.success(f"Loaded {len(df_up):,} rows from file.")
            sample_idx = st.sidebar.slider("Row Index", 0, len(df_up)-1, 0)
            sample_input = df_up.iloc[sample_idx].to_dict()
        else:
            sample_input = preset_dict["Benign IoT Sensor Flow"]

    if sample_input:
        res = engine.predict_flow(sample_input)
        shap_res = explainer.explain_flow(sample_input, predicted_class_name=res['supervised_label'])

        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("### 🚨 Live Detection Result")
            badge_text = res['final_badge']
            if badge_text == "BENIGN":
                st.markdown(f'<div class="badge-benign">🟢 BENIGN TRAFFIC</div>', unsafe_allow_html=True)
            elif badge_text == "POTENTIAL ZERO-DAY ANOMALY":
                st.markdown(f'<div class="badge-zeroday">🟠 POTENTIAL ZERO-DAY ANOMALY</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-attack">🔴 ATTACK DETECTED: {badge_text}</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Inference Latency", f"{res['latency_ms']:.2f} ms")
            with m2:
                st.metric("AE Reconstruction MSE", f"{res['reconstruction_mse']:.5f}")
            with m3:
                st.metric("AE Threshold (p99)", f"{res['reconstruction_threshold']:.5f}")

            st.markdown("---")
            st.markdown("#### 🧠 Classification Rationale")
            st.info(shap_res['rationale'])

            st.markdown("#### 📊 Supervised Probability Breakdown")
            prob_df = pd.DataFrame(list(res['class_probabilities'].items()), columns=['Class', 'Probability'])
            prob_df = prob_df.sort_values(by='Probability', ascending=False).head(5)
            st.dataframe(prob_df, width="stretch", hide_index=True)

        with col_right:
            st.markdown("### 🔍 Real-Time SHAP Local Feature Attribution")
            fig = explainer.generate_waterfall_plot(shap_res)
            st.pyplot(fig)
            plt.close()

            log_entry = {
                "Timestamp": time.strftime("%H:%M:%S"),
                "Final Decision": res['final_badge'],
                "Classification Type": res['classification_type'],
                "Reconstruction MSE": f"{res['reconstruction_mse']:.5f}",
                "Confidence": f"{res['supervised_confidence']*100:.1f}%",
                "Latency (ms)": f"{res['latency_ms']:.2f}",
                "Top Driver": shap_res['top_drivers'][0]['feature'] if shap_res['top_drivers'] else "N/A"
            }
            if not st.session_state.alert_log or st.session_state.alert_log[-1] != log_entry:
                st.session_state.alert_log.append(log_entry)

        st.markdown("---")
        st.markdown("### 📜 Security Operations Alert Log & Audit Export")
        log_df = pd.DataFrame(st.session_state.alert_log)
        st.dataframe(log_df, width="stretch")

        csv_data = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Security Audit Log (CSV)",
            data=csv_data,
            file_name="iot_ids_security_audit_log.csv",
            mime="text/csv"
        )
