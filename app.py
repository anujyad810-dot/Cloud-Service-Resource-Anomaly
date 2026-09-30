import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cloud Anomaly Detector", page_icon="☁️", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load('cloud_anomaly_model.pkl')
    scaler = joblib.load('cloud_scaler.pkl')
    try:
        threshold = joblib.load('best_threshold.pkl')
    except Exception:
        threshold = 0.50
    return model, scaler, threshold

model, scaler, optimal_threshold = load_assets()

st.title("☁️ Cloud Server Anomaly Detection")
st.caption("Classical Machine Learning Telemetry Monitoring Pipeline")

st.sidebar.header("🎛️ Live Server Metrics")
cpu = st.sidebar.slider("CPU Usage (Normalized)", 0.0, 1.0, 0.25, step=0.01)
memory = st.sidebar.slider("Memory Usage (Normalized)", 0.0, 1.0, 0.30, step=0.01)
request_rate = st.sidebar.slider("Request Rate (Normalized)", 0.0, 1.0, 0.20, step=0.01)
latency = st.sidebar.slider("Latency (Normalized)", 0.0, 1.0, 0.15, step=0.01)
throughput = st.sidebar.slider("Network Throughput (Normalized)", 0.0, 1.0, 0.30, step=0.01)

if st.button("Run System Diagnostics", use_container_width=True):
    # Calculate all potential feature combinations
    feature_dict = {
        'cpu_usage': cpu,
        'memory_usage': memory,
        'request_rate': request_rate,
        'latency': latency,
        'network_throughput': throughput,
        'system_load': cpu * memory,
        'latency_per_req': latency / (request_rate + 0.001),
        'throughput_efficiency': throughput / (cpu + 0.001),
        'resource_stress': np.sqrt(cpu**2 + memory**2)
    }
    
    # Automatically detect expected features from the loaded scaler
    if hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
    else:
        # Fallback to standard 8-feature configuration
        expected_cols = [
            'cpu_usage', 'memory_usage', 'request_rate', 'latency', 'network_throughput',
            'system_load', 'latency_per_req', 'throughput_efficiency'
        ]
    
    # Filter inputs to match loaded scaler features exactly
    input_data = [[feature_dict[col] for col in expected_cols]]
    input_df = pd.DataFrame(input_data, columns=expected_cols)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Make Prediction
    anomaly_prob = model.predict_proba(input_scaled)[0][1]
    is_anomaly = int(anomaly_prob >= optimal_threshold)
    
    st.markdown("---")
    st.subheader("📊 Diagnostic Summary")
    
    col1, col2 = st.columns(2)
    col1.metric("Anomaly Probability", f"{anomaly_prob * 100:.2f}%")
    col2.metric("Decision Threshold", f"{optimal_threshold * 100:.2f}%")
    
    if is_anomaly == 1:
        st.error(f"🚨 **ANOMALY DETECTED!** Risk Score: {anomaly_prob*100:.1f}%")
        st.warning("Server telemetry exceeded decision boundaries.")
    else:
        st.success(f"✅ **SYSTEM HEALTHY.** Risk Score: {anomaly_prob*100:.1f}%")
        st.info("Metrics operating within safe operational ranges.")