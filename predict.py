import joblib
import pandas as pd
import json
from ai_service import GROQ_MODEL, generate_ai_response


def get_fault_type(parameters):
    """Use Groq to identify likely fault conditions from ML results."""
    prompt = f"""
You are an industrial motor diagnostic assistant.

Analyze the motor measurements and machine-learning results below.
Return a concise fault assessment with:
1. likely fault type or types
2. evidence from the supplied parameters
3. confidence (low, medium, or high)

Use only the supplied data. Treat the result as a possible fault, not a
confirmed mechanical diagnosis. If the data is insufficient, say so.

Motor data:
{json.dumps(parameters, indent=2)}
"""

    try:
        return generate_ai_response(prompt)
    except Exception as error:
        return f"Groq diagnosis unavailable: {error}"


def get_rag_instructions(fault_type):
    """Retrieve approved procedures and ask Groq to explain them."""
    try:
        import sys
        from pathlib import Path

        rag_dir = Path(__file__).resolve().parent / "motor-rag"
        if str(rag_dir) not in sys.path:
            sys.path.insert(0, str(rag_dir))

        from rag import retrieve_procedure
        from groq_service import generate_instruction

        results = retrieve_procedure(fault_type, top_k=3)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        if not documents:
            return "No approved maintenance procedure was found.", []

        context_parts = []
        sources = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", -1)
            location = f"{source}, page {page}" if page != -1 else source
            context_parts.append(f"SOURCE: {location}\n{document}")
            sources.append(location)

        context = "\n\n".join(context_parts)
        return generate_instruction(fault_type, context), sources
    except Exception as error:
        return f"RAG procedure retrieval unavailable: {error}", []


# ============================================================
# 1. LOAD TRAINED MODELS
# ============================================================

package = joblib.load(
    "industrial_motor_ai_models.joblib"
)

model = package["health_model"]

anomaly_model = package["anomaly_model"]

# IMPORTANT:
# Isolation Forest was trained using scaled data,
# so we must load the same scaler used during training.
scaler = package["anomaly_scaler"]

features = package["features"]

class_names = package["class_names"]

health_map = package["health_map"]

# Reference limits saved during training.  Isolation Forest scores are lower
# for more unusual observations.
score_low = package.get("anomaly_score_low", -0.2)
score_high = package.get("anomaly_score_high", 0.2)


# ============================================================
# 2. ENTER MOTOR SENSOR VALUES
# ============================================================

print("\n" + "=" * 60)
print("ENTER MOTOR SENSOR VALUES")
print("=" * 60)

voltage = float(
    input("Voltage (V): ")
)

current = float(
    input("Current (A): ")
)

temperature = float(
    input("Temperature (°C): ")
)

vibration = float(
    input("Vibration: ")
)


# ============================================================
# 3. DERIVED FEATURES
# ============================================================

power = (
    voltage * current
)

thermal_load = (
    temperature * current
)


# ============================================================
# 4. CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([{

    "voltage_v": voltage,

    "current_a": current,

    "temperature_c": temperature,

    "vibration": vibration,

    "power_w": power,

    "thermal_load": thermal_load

}])

# Make sure the feature order is exactly the same
# as the order used during model training.

input_data = input_data[features]


# ============================================================
# 5. HEALTH PREDICTION
# ============================================================

prediction = model.predict(
    input_data
)[0]

probabilities = model.predict_proba(
    input_data
)[0]


# ============================================================
# 6. CONFIDENCE
# ============================================================

confidence = max(
    probabilities
)


# ============================================================
# 7. HEALTH INDEX
# ============================================================

health_index = health_map[
    prediction
]

# Probability-weighted score gives a smoother result than the fixed class
# lookup alone.  It reflects uncertainty between neighboring health classes.
health_score = sum(
    probabilities[i] * health_map[i]
    for i in range(len(probabilities))
)


# ============================================================
# 8. ANOMALY DETECTION
# ============================================================

# Scale input using the SAME scaler used during training.

scaled_input = scaler.transform(
    input_data
)

# Preserve feature names.

scaled_input = pd.DataFrame(
    scaled_input,
    columns=features
)

anomaly_prediction = anomaly_model.predict(
    scaled_input
)[0]

anomaly_score = float(anomaly_model.decision_function(scaled_input)[0])

# Map the model score to an easy-to-read 0-100 anomaly risk.  Clamp values so
# unusual future readings cannot produce percentages outside the range.
anomaly_risk = 100 * (score_high - anomaly_score) / (score_high - score_low)
anomaly_risk = max(0.0, min(100.0, anomaly_risk))

is_anomaly = (
    anomaly_prediction == -1
)

all_parameters = {
    "voltage_v": voltage,
    "current_a": current,
    "temperature_c": temperature,
    "vibration": vibration,
    "power_w": power,
    "thermal_load": thermal_load,
    "health_state": class_names[prediction],
    "health_code": int(prediction),
    "health_index": health_index,
    "health_score": round(float(health_score), 2),
    "classification_confidence": round(float(confidence), 4),
    "anomaly_score": round(anomaly_score, 4),
    "anomaly_risk": round(anomaly_risk, 2),
    "anomaly_detected": bool(is_anomaly)
}

all_parameters["class_probabilities"] = {
    class_names[i]: round(float(probabilities[i]), 4)
    for i in range(len(probabilities))
}
# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("       INDUSTRIAL MOTOR ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# INPUT VALUES
# ------------------------------------------------------------

print("\nInput Sensor Values")
print("-" * 60)

print(
    f"Voltage      : {voltage:.2f} V"
)

print(
    f"Current      : {current:.2f} A"
)

print(
    f"Temperature  : {temperature:.2f} °C"
)

print(
    f"Vibration    : {vibration:.2f}"
)


# ------------------------------------------------------------
# DERIVED VALUES
# ------------------------------------------------------------

print("\nDerived Features")
print("-" * 60)

print(
    f"Power        : {power:.2f} W"
)

print(
    f"Thermal Load : {thermal_load:.2f}"
)


# ------------------------------------------------------------
# HEALTH RESULT
# ------------------------------------------------------------

print("\nMotor Health")
print("-" * 60)

print(
    f"Health State : "
    f"{class_names[prediction]}"
)

print(
    f"Health Code  : "
    f"{prediction}"
)

print(
    f"Health Index : "
    f"{health_index}%"
)

print(
    f"Health Score : "
    f"{health_score:.1f}%"
)

print(
    f"Confidence   : "
    f"{confidence * 100:.2f}%"
)


# ------------------------------------------------------------
# PROBABILITIES
# ------------------------------------------------------------

print("\nPrediction Probabilities")
print("-" * 60)

for i, probability in enumerate(probabilities):

    print(
        f"{class_names[i]:10} : "
        f"{probability * 100:.2f}%"
    )


# ------------------------------------------------------------
# ANOMALY
# ------------------------------------------------------------

print("\nAnomaly Detection")
print("-" * 60)

if is_anomaly:

    print(
        "WARNING: Abnormal sensor pattern detected!"
    )

else:

    print(
        "No abnormal sensor pattern detected."
    )

print(f"Anomaly Score: {anomaly_score:.4f}")
print(f"Anomaly Risk : {anomaly_risk:.1f}%")


# ------------------------------------------------------------
# AI FAULT TYPE
# ------------------------------------------------------------

print(f"\nAI Fault Assessment (Groq: {GROQ_MODEL})")
print("-" * 60)
print("The AI reports likely causes; it cannot confirm a mechanical fault.")
fault_type = get_fault_type(all_parameters)
print(fault_type)


# ------------------------------------------------------------
# RAG MAINTENANCE PROCEDURE
# ------------------------------------------------------------

print("\nApproved Maintenance Procedure (RAG)")
print("-" * 60)
procedure, procedure_sources = get_rag_instructions(fault_type)
print(procedure)
if procedure_sources:
    print("\nProcedure sources:")
    for source in procedure_sources:
        print(f"- {source}")


# ------------------------------------------------------------
# COMPLETE PARAMETER SUMMARY
# ------------------------------------------------------------

print("\nAll Calculated Parameters")
print("-" * 60)
for parameter, value in all_parameters.items():
    print(f"{parameter:30}: {value}")


# ============================================================
# 10. FINAL RECOMMENDATION
# ============================================================

print("\nMotor Condition")
print("-" * 60)

if health_score >= 80 and anomaly_risk < 50:

    print(
        "STATUS: HEALTHY"
    )

    print(
        "Recommendation: Continue normal operation."
    )


elif health_score >= 60 and anomaly_risk < 70:

    print(
        "STATUS: LOW HEALTH"
    )

    print(
        "Recommendation: Monitor motor condition."
    )


elif health_score >= 40 and anomaly_risk < 85:

    print(
        "STATUS: MODERATE"
    )

    print(
        "Recommendation: Schedule inspection."
    )


else:

    print(
        "STATUS: HIGH RISK"
    )

    print(
        "Recommendation: Immediate inspection recommended."
    )


# ------------------------------------------------------------
# ADDITIONAL ANOMALY ALERT
# ------------------------------------------------------------

if is_anomaly:

    print(
        "\nAdditional Alert:"
    )

    print(
        "Sensor readings show an unusual pattern."
    )

    print(
        "Check the motor and sensor hardware."
    )


# ============================================================
# 11. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)
