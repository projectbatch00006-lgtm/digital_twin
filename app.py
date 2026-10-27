"""MotorGuard Streamlit dashboard."""
import base64, json, os, sys
from datetime import datetime
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
BLEND = ROOT / "digital_twin" / "Electric_Motor.blend"
GLB = ROOT / "digital_twin" / "Electric_Motor.glb"
MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
URL = "http://localhost:11434/api/chat"

st.set_page_config(page_title="MotorGuard Digital Twin", page_icon="⚙️", layout="wide")
st.markdown("""<style>.block-container{max-width:1500px;padding-top:1.5rem}.hero{padding:1.4rem;border-radius:16px;background:linear-gradient(120deg,#10233e,#173b58);color:white;margin-bottom:1rem}.hero h1{margin:0}.muted{color:#94a3b8;font-size:.85rem}@media(max-width:700px){.block-container{padding:.7rem}}</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return joblib.load(MODEL_FILE)

def ask(prompt):
    body = json.dumps({"model": MODEL, "stream": False, "messages": [{"role": "user", "content": prompt}]}).encode()
    try:
        with urlopen(Request(URL, data=body, headers={"Content-Type": "application/json"}, method="POST"), timeout=120) as response:
            return json.loads(response.read().decode()).get("message", {}).get("content", "No response returned.")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return f"Mistral unavailable: {error}"

def run_rag(fault):
    if str(RAG_DIR) not in sys.path: sys.path.insert(0, str(RAG_DIR))
    try:
        from rag import retrieve_procedure
        from ollama_service import generate_instruction
        found = retrieve_procedure(fault, top_k=3)
        docs, metas = found.get("documents", [[]])[0], found.get("metadatas", [[]])[0]
        if not docs: return "No approved maintenance procedure was found.", []
        context, sources = [], []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            source, page = meta.get("source", "Unknown source"), meta.get("page", -1)
            location = f"{source}, page {page}" if page != -1 else source
            context.append(f"SOURCE: {location}\n{doc}"); sources.append(location)
        return generate_instruction(fault, "\n\n".join(context)), sources
    except Exception as error:
        return f"RAG unavailable: {error}", []

def diagnose(values, package):
    features = package["features"]; frame = pd.DataFrame([values])[features]
    model = package["health_model"]; prediction = int(model.predict(frame)[0]); probs = model.predict_proba(frame)[0]
    health_map = package["health_map"]; scaled = pd.DataFrame(package["anomaly_scaler"].transform(frame), columns=features)
    anomaly = package["anomaly_model"]; label = int(anomaly.predict(scaled)[0]); score = float(anomaly.decision_function(scaled)[0])
    low, high = package.get("anomaly_score_low", -.2), package.get("anomaly_score_high", .2)
    risk = max(0, min(100, 100 * (high-score) / (high-low)))
    return {**values, "health_state": package["class_names"][prediction], "health_code": prediction,
        "health_index": health_map[prediction], "health_score": round(sum(probs[i]*health_map[i] for i in range(len(probs))), 2),
        "classification_confidence": round(float(max(probs)), 4), "class_probabilities": {package["class_names"][i]: round(float(probs[i]),4) for i in range(len(probs))},
        "anomaly_score": round(score,4), "anomaly_risk": round(float(risk),2), "anomaly_detected": label == -1}

def status(result):
    if result["health_score"] >= 80 and result["anomaly_risk"] < 50: return "HEALTHY", "Continue normal monitoring."
    if result["health_score"] >= 60 and result["anomaly_risk"] < 70: return "MONITOR", "Review trends and continue monitoring."
    if result["health_score"] >= 40 and result["anomaly_risk"] < 85: return "INSPECTION REQUIRED", "Schedule a qualified inspection."
    return "HIGH RISK", "Stop according to site procedure and escalate."

def show_3d():
    if not GLB.exists(): st.info("Export Electric_Motor.blend as digital_twin/Electric_Motor.glb to enable the viewer."); st.code(str(BLEND)); return
    encoded = base64.b64encode(GLB.read_bytes()).decode("ascii")
    components.html(f'''<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script><model-viewer src="data:model/gltf-binary;base64,{encoded}" camera-controls auto-rotate ar shadow-intensity="1" style="width:100%;height:600px;background:#0b1220;border-radius:16px"></model-viewer>''', height=620)

st.markdown('<div class="hero"><h1>⚙️ MotorGuard Digital Twin</h1><p>Condition monitoring • anomaly detection • approved procedures • 3D motor review</p></div>', unsafe_allow_html=True)
package = load_models()
if "history" not in st.session_state: st.session_state.history = []
with st.sidebar:
    st.header("Sensor Input")
    with st.form("sensor_form"):
        voltage = st.number_input("Voltage (V)", 0.0, 1000.0, 400.0)
        current = st.number_input("Current (A)", 0.0, 200.0, 15.0)
        temperature = st.number_input("Temperature (°C)", 0.0, 250.0, 40.0)
        vibration = st.number_input("Vibration", 0.0, 100.0, 1.5)
        run = st.form_submit_button("Run Diagnosis", type="primary", use_container_width=True)
    st.caption(f"Model: {MODEL}"); st.caption("AI output is advisory; approved procedures and qualified personnel remain authoritative.")
if run:
    values = {"voltage_v": voltage, "current_a": current, "temperature_c": temperature, "vibration": vibration, "power_w": voltage*current, "thermal_load": temperature*current}
    result = diagnose(values, package)
    with st.spinner("Analyzing health and anomaly..."): fault = ask("Identify likely motor fault types from these ML results. Give evidence and confidence; do not claim confirmation:\n" + json.dumps(result, indent=2))
    with st.spinner("Retrieving approved maintenance procedure..."): procedure, sources = run_rag(fault)
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.update(result=result, fault=fault, procedure=procedure, sources=sources)
    st.session_state.history.append({"timestamp":result["timestamp"],"health":result["health_state"],"score":result["health_score"],"anomaly_risk":result["anomaly_risk"]})
if "result" not in st.session_state:
    st.info("Enter sensor readings and select Run Diagnosis.")
else:
    result = st.session_state.result; label, action = status(result)
    st.warning(f"{label}: {action}") if label != "HEALTHY" else st.success(f"{label}: {action}")
    overview, charts, procedures, twin, history = st.tabs(["Overview", "Parameters & Charts", "Approved Procedure", "3D Digital Twin", "History"])
    with overview:
        c1,c2,c3,c4 = st.columns(4); c1.metric("Health Score",f"{result['health_score']:.1f}%"); c2.metric("Health State",result["health_state"]); c3.metric("Anomaly Risk",f"{result['anomaly_risk']:.1f}%"); c4.metric("Confidence",f"{result['classification_confidence']*100:.1f}%")
        st.subheader("Likely Fault Assessment"); st.write(st.session_state.fault); st.caption("Likely condition only; confirmation requires qualified inspection.")
        st.error("Abnormal sensor pattern detected") if result["anomaly_detected"] else st.success("No anomaly flag detected")
    with charts:
        st.subheader("Health probabilities"); st.bar_chart(pd.DataFrame({"Probability":result["class_probabilities"]}))
        st.subheader("All calculated parameters"); st.dataframe(pd.DataFrame([{k:v for k,v in result.items() if k != "class_probabilities"}]).T.rename(columns={0:"Value"}), use_container_width=True); st.json(result["class_probabilities"])
    with procedures:
        st.subheader("Approved Maintenance Procedure"); st.write(st.session_state.procedure)
        if st.session_state.sources: st.subheader("Sources"); [st.write(f"• {source}") for source in st.session_state.sources]
        report = json.dumps({"diagnosis":result,"fault":st.session_state.fault,"procedure":st.session_state.procedure,"sources":st.session_state.sources}, indent=2)
        st.download_button("Download diagnosis report", report, "motor_diagnosis.json", "application/json")
    with twin: show_3d()
    with history: st.subheader("Recent diagnoses"); st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
