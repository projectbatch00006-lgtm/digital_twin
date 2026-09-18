import joblib
import pandas as pd


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

is_anomaly = (
    anomaly_prediction == -1
)
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


# ============================================================
# 10. FINAL RECOMMENDATION
# ============================================================

print("\nMotor Condition")
print("-" * 60)

if prediction == 0 and not is_anomaly:

    print(
        "STATUS: HEALTHY"
    )

    print(
        "Recommendation: Continue normal operation."
    )


elif prediction == 1:

    print(
        "STATUS: LOW HEALTH"
    )

    print(
        "Recommendation: Monitor motor condition."
    )


elif prediction == 2:

    print(
        "STATUS: MODERATE"
    )

    print(
        "Recommendation: Schedule inspection."
    )


elif prediction == 3:

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