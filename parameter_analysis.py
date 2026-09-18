import joblib
import shap
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load trained model
# --------------------------------------------------

print("Loading model...")

models = joblib.load("industrial_motor_ai_models.joblib")

health_model = models["health_model"]
features = models["features"]
class_names = models["class_names"]

print("Model loaded successfully!")
print("Features:", features)
print("Classes:", class_names)


# --------------------------------------------------
# 2. Create SHAP explainer
# --------------------------------------------------

explainer = shap.TreeExplainer(health_model)

print("SHAP explainer created successfully!")


# --------------------------------------------------
# 3. Example motor input
# --------------------------------------------------

motor_input = {
    "voltage_v": 238,
    "current_a": 8.2,
    "temperature_c": 91,
    "vibration": 1.7,
    "power_w": 238 * 8.2,
    "thermal_load": 91 * 8.2
}

X_motor = pd.DataFrame([motor_input])

# Make sure feature order is exactly the same as training
X_motor = X_motor[features]


# --------------------------------------------------
# 4. Random Forest prediction
# --------------------------------------------------

prediction = health_model.predict(X_motor)[0]

probabilities = health_model.predict_proba(X_motor)[0]

print("\n==============================")
print("MOTOR HEALTH PREDICTION")
print("==============================")

print("Predicted class:", prediction)

if isinstance(class_names, dict):
    print("Health state:", class_names[prediction])
else:
    print("Health state:", class_names[prediction])

print("\nClass probabilities:")

for i, probability in enumerate(probabilities):
    print(f"{i}: {probability:.4f}")


# --------------------------------------------------
# 5. Calculate SHAP values
# --------------------------------------------------

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(X_motor)

print("SHAP values calculated successfully!")


# --------------------------------------------------
# 6. Display SHAP values
# --------------------------------------------------

print("\n==============================")
print("PARAMETER CONTRIBUTIONS")
print("==============================")

print("Input values:")

for feature in features:
    print(f"{feature}: {X_motor.iloc[0][feature]}")


print("\nRaw SHAP output shape:")

if isinstance(shap_values, list):
    print([np.array(x).shape for x in shap_values])
else:
    print(np.array(shap_values).shape)

    # --------------------------------------------------
# 7. Explain the predicted class
# --------------------------------------------------

# SHAP 0.52.0 output for this model:
# Shape = (samples, features, classes)

shap_array = np.array(shap_values)

predicted_class = int(prediction)

# Extract SHAP values for the predicted class
class_shap_values = shap_array[0, :, predicted_class]

# Create parameter contribution table
contribution_df = pd.DataFrame({
    "Parameter": features,
    "Value": X_motor.iloc[0].values,
    "SHAP_Value": class_shap_values,
    "Absolute_Impact": np.abs(class_shap_values)
})

# Sort by importance
contribution_df = contribution_df.sort_values(
    by="Absolute_Impact",
    ascending=False
)

print("\n==============================")
print("SHAP PARAMETER ANALYSIS")
print("==============================")

print(f"Predicted Health: {class_names[predicted_class]}")

print("\nParameter contributions:")

for _, row in contribution_df.iterrows():

    direction = "↑" if row["SHAP_Value"] > 0 else "↓"

    print(
        f"{row['Parameter']:15s} "
        f"Value={row['Value']:8.2f}   "
        f"SHAP={row['SHAP_Value']:+.4f}   "
        f"{direction}"
    )