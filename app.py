"""Streamlit dashboard for the motor-health, anomaly, RAG, and 3D workflow."""

import base64
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "industrial_motor_ai_models.joblib"
RAG_DIR = ROOT / "motor-rag"
BLENDER_FILE = ROOT / "digital_twin" / "Electric_Motor.blend"
WEB_MODEL_FILE = ROOT / "digital_twin" / "Electric_Motor.glb"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")


@st.cache_resource
def load_models():
    package = joblib.load(MODEL_FILE)
    return package


def ask_mistral(prompt):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result.get("message", {}).get("content", "No response returned.")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return f"Mistral is unavailable: {error}"


def run_rag(fault_type):
    """Retrieve approved procedure chunks and ask Mistral to format them."""
    if str(RAG_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_DIR))
    try:
        from rag import retrieve_procedure
        from ollama_service import generate_instruction

        results = retrieve_procedure(fault_type, top_k=3)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        if not documents:
            return "No approved procedure was found.", []
        context = []
        sources = []
        for i, document in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", -1)
            location = f"{source}, page {page}" if page != -1 else source
            context.append(f"SOURCE: {location}\n{document}")
            sources.append(location)
        return generate_instruction(fault_type, "\n\n".join(context)), sources
    except Exception as error:
        return f"RAG is unavailable: {error}", []


def diagnosis(parameters, package):
    features = package["features"]
    frame = pd.DataFrame([parameters])[features]
    health_model = package["health_model"]
    prediction = int(health_model.predict(frame)[0])
    probabilities = health_model.predict_proba(frame)[0]
    health_map = package["health_map"]
    health_score = sum(probabilities[i] * health_map[i] for i in range(len(probabilities)))

    scaler = package["anomaly_scaler"]
    anomaly_model = package["anomaly_model"]
    scaled = pd.DataFrame(scaler.transform(frame), columns=features)
    anomaly_label = int(anomaly_model.predict(scaled)[0])
    anomaly_score = float(anomaly_model.decision_function(scaled)[0])
    low = package.get("anomaly_score_low", -0.2)
    high = package.get("anomaly_score_high", 0.2)
    anomaly_risk = max(0, min(100, 100 * (high - anomaly_score) / (high - low)))

    result = {
        **parameters,
        "health_state": package["class_names"][prediction],
        "health_code": prediction,
        "health_index": health_map[prediction],
        "health_score": round(float(health_score), 2),
        "classification_confidence": round(float(max(probabilities)), 4),
        "class_probabilities": {
            package["class_names"][i]: round(float(probabilities[i]), 4)
            for i in range(len(probabilities))
        },
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_risk": round(float(anomaly_risk), 2),
        "anomaly_detected": anomaly_label == -1
    }
    return result


def show_3d_viewer():
    st.subheader("3D Motor / Digital Twin")
    if not WEB_MODEL_FILE.exists():
        st.info(
            "The Blender file is present, but browsers cannot display .blend files directly. "
            "Export digital_twin/Electric_Motor.blend to GLB or glTF as "
            "digital_twin/Electric_Motor.glb to enable the interactive viewer."
        )
        st.code(str(BLENDER_FILE))
        return
    # Embed the local GLB in the component. Browser iframes cannot reliably
    # fetch a file:// URL from a Streamlit page.
    model_data = base64.b64encode(WEB_MODEL_FILE.read_bytes()).decode("ascii")
    model_uri = f"data:model/gltf-binary;base64,{model_data}"
    components.html(
        f"""<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <model-viewer src="{model_uri}" camera-controls auto-rotate shadow-intensity="1"
          style="width:100%;height:600px;background:#111;border-radius:12px;">
        </model-viewer>""",
        height=620,
    )


st.set_page_config(page_title="Industrial Motor Digital Twin", page_icon="⚙️", layout="wide")
st.title("Industrial Motor Health & Digital Twin")
st.caption("Random Forest + Isolation Forest + approved-procedure RAG + Blender 3D asset")

package = load_models()
with st.sidebar:
    st.header("Motor Sensor Inputs")
    voltage = st.number_input("Voltage (V)", min_value=0.0, value=400.0)
    current = st.number_input("Current (A)", min_value=0.0, value=15.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, value=40.0)
    vibration = st.number_input("Vibration", min_value=0.0, value=1.5)
    run = st.button("Run Diagnosis", type="primary", use_container_width=True)

if run:
    parameters = {
        "voltage_v": voltage,
        "current_a": current,
        "temperature_c": temperature,
        "vibration": vibration,
        "power_w": voltage * current,
        "thermal_load": temperature * current,
    }
    result = diagnosis(parameters, package)
    st.session_state["diagnosis"] = result
    with st.spinner("Mistral is identifying likely fault conditions..."):
        fault_prompt = (
            "Identify the likely motor fault type from these verified ML outputs. "
            "Give evidence and confidence, but do not claim confirmation:\n" +
            json.dumps(result, indent=2)
        )
        st.session_state["fault_type"] = ask_mistral(fault_prompt)
    with st.spinner("Retrieving approved maintenance procedures..."):
        procedure, sources = run_rag(st.session_state["fault_type"])
        st.session_state["procedure"] = procedure
        st.session_state["sources"] = sources

if "diagnosis" not in st.session_state:
    st.info("Enter sensor values in the sidebar and select Run Diagnosis.")
else:
    result = st.session_state["diagnosis"]
    overview, parameters_tab, procedure_tab, twin_tab = st.tabs(
        ["Diagnosis", "All Parameters", "Maintenance Procedure", "3D Digital Twin"]
    )
    with overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health State", result["health_state"])
        c2.metric("Health Score", f"{result['health_score']:.1f}%")
        c3.metric("Anomaly Risk", f"{result['anomaly_risk']:.1f}%")
        c4.metric("Confidence", f"{result['classification_confidence'] * 100:.1f}%")
        st.subheader("Likely Fault Type (Mistral)")
        st.write(st.session_state.get("fault_type", "Unavailable"))
        if result["anomaly_detected"]:
            st.error("Abnormal sensor pattern detected")
        else:
            st.success("No Isolation Forest anomaly flag detected")
    with parameters_tab:
        st.json(result)
        st.dataframe(pd.DataFrame([result]), use_container_width=True)
    with procedure_tab:
        st.subheader("Approved Maintenance Procedure")
        st.write(st.session_state.get("procedure", "Unavailable"))
        sources = st.session_state.get("sources", [])
        if sources:
            st.subheader("Sources")
            for source in sources:
                st.write(f"- {source}")
    with twin_tab:
        show_3d_viewer()
