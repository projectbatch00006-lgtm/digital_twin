import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "data/industrial_motor_cleaned.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("INDUSTRIAL MOTOR - AI HEALTH MONITORING")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. CHECK DATA
# ============================================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nHealth State Distribution:")
print(df["health_state"].value_counts())

print("\nHealth Code Distribution:")
print(
    df["health_code"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

features = [
    "voltage_v",
    "current_a",
    "temperature_c",
    "vibration",
    "power_w",
    "thermal_load"
]

target = "health_code"

X = df[features]
y = df[target]
# ============================================================
# 3A. SCALING FOR ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 60)
print("SCALING DATA FOR ANOMALY DETECTION")
print("=" * 60)

scaler = StandardScaler()

X_anomaly = scaler.fit_transform(X)

# Convert scaled array to DataFrame
# and preserve feature names.

X_anomaly = pd.DataFrame(
    X_anomaly,
    columns=features,
    index=X.index
)

print("Feature scaling completed.")
# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Total samples :", len(df))
print("Training      :", len(X_train))
print("Testing       :", len(X_test))


# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=1# changed from -1 to 1 This disables multiprocessing and should stop those repeated parallel warnings. Training may be slightly slower, but with only 8,000 rows, that shouldn't be a problem for your project.
)


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 7. TEST PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 9. CROSS VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1_weighted"
)

print("\nF1 scores:")

for i, score in enumerate(cv_scores, 1):

    print(
        f"Fold {i}: {score:.4f}"
    )

print(
    f"\nMean F1 Score: "
    f"{cv_scores.mean():.4f}"
)

print(
    f"Standard Deviation: "
    f"{cv_scores.std():.4f}"
)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

class_names = [
    "Normal",
    "Low",
    "Moderate",
    "High"
]

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title(
    "Random Forest - Motor Health Classification"
)

plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png",
    dpi=300
)

#plt.show()---> commented due to some execution errors 


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance)


plt.figure(
    figsize=(8, 5)
)

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "random_forest_feature_importance.png",
    dpi=300
)

#plt.show()---> runtime errors 


# ============================================================
# 13. ANOMALY DETECTION - PARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("ANOMALY DETECTION - PARAMETER TUNING")
print("=" * 60)


# Actual anomaly labels

y_actual = df["outlier_flag"].astype(int)


# Finer tuning near previous best region

contamination_values = [
    0.17,
    0.18,
    0.183,
    0.185,
    0.187,
    0.19,
    0.194,
    0.20,
    0.21
]


best_f1 = 0
best_contamination = None

print("\nTesting different contamination values...")
print("-" * 75)


for contamination in contamination_values:

    test_model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=1# changed from -1 to 1 This disables multiprocessing and should stop those repeated parallel warnings. Training may be slightly slower, but with only 8,000 rows, that shouldn't be a problem for your project.

    )

    test_prediction = test_model.fit_predict(
        X_anomaly
    )


    # Isolation Forest:
    # -1 = anomaly
    #  1 = normal

    y_test_anomaly = (
        test_prediction == -1
    ).astype(int)


    test_accuracy = accuracy_score(
        y_actual,
        y_test_anomaly
    )

    test_precision = precision_score(
        y_actual,
        y_test_anomaly,
        zero_division=0
    )

    test_recall = recall_score(
        y_actual,
        y_test_anomaly,
        zero_division=0
    )

    test_f1 = f1_score(
        y_actual,
        y_test_anomaly,
        zero_division=0
    )


    print(
        f"Contamination: {contamination:.3f} | "
        f"Accuracy: {test_accuracy:.4f} | "
        f"Precision: {test_precision:.4f} | "
        f"Recall: {test_recall:.4f} | "
        f"F1: {test_f1:.4f}"
    )


    # Store best contamination

    if test_f1 > best_f1:

        best_f1 = test_f1
        best_contamination = contamination


# ============================================================
# 14. TRAIN FINAL ANOMALY MODEL
# ============================================================

print("\n" + "=" * 60)
print("BEST ANOMALY MODEL")
print("=" * 60)

print(
    f"Best contamination : "
    f"{best_contamination:.3f}"
)

print(
    f"Best F1 Score      : "
    f"{best_f1:.4f}"
)


anomaly_model = IsolationForest(
    n_estimators=300,
    contamination=best_contamination,
    random_state=42,
    n_jobs=-1
)


# Train with scaled features

anomaly_model.fit(
    X_anomaly
)


# Predict entire dataset

anomaly_prediction = anomaly_model.predict(
    X_anomaly
)


df["predicted_outlier"] = (
    anomaly_prediction == -1
)


# ============================================================
# 14A. FINAL ANOMALY PERFORMANCE
# ============================================================

y_anomaly = (
    anomaly_prediction == -1
).astype(int)


final_accuracy = accuracy_score(
    y_actual,
    y_anomaly
)

final_precision = precision_score(
    y_actual,
    y_anomaly,
    zero_division=0
)

final_recall = recall_score(
    y_actual,
    y_anomaly,
    zero_division=0
)

final_f1 = f1_score(
    y_actual,
    y_anomaly,
    zero_division=0
)


print("\n" + "=" * 60)
print("FINAL ANOMALY DETECTION PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy  : {final_accuracy:.4f}"
)

print(
    f"Precision : {final_precision:.4f}"
)

print(
    f"Recall    : {final_recall:.4f}"
)

print(
    f"F1 Score  : {final_f1:.4f}"
)


# ============================================================
# 14B. PREDICTED ANOMALY COUNT
# ============================================================

print("\nPredicted anomalies:")

print(
    df["predicted_outlier"]
    .value_counts()
)


# ============================================================
# 14C. ANOMALY CONFUSION MATRIX
# ============================================================

anomaly_cm = confusion_matrix(
    y_actual,
    y_anomaly
)

print("\n" + "=" * 60)
print("ANOMALY CONFUSION MATRIX")
print("=" * 60)

print(anomaly_cm)


anomaly_disp = ConfusionMatrixDisplay(
    confusion_matrix=anomaly_cm,
    display_labels=[
        "Normal",
        "Outlier"
    ]
)

anomaly_disp.plot()

plt.title(
    "Isolation Forest - Anomaly Detection"
)

plt.tight_layout()

plt.savefig(
    "isolation_forest_confusion_matrix.png",
    dpi=300
)

#plt.show()---> runtime error 


# ============================================================
# 14D. COMPARE WITH EXISTING OUTLIER FLAG
# ============================================================

if "outlier_flag" in df.columns:

    print("\n" + "=" * 60)
    print("ANOMALY COMPARISON")
    print("=" * 60)

    comparison = pd.crosstab(
        df["outlier_flag"],
        df["predicted_outlier"],
        rownames=["Actual Outlier"],
        colnames=["Predicted Outlier"]
    )

    print(comparison)


# ============================================================
# 15. HEALTH INDEX
# ============================================================

print("\n" + "=" * 60)
print("HEALTH INDEX")
print("=" * 60)

health_map = {
    0: 100,
    1: 75,
    2: 50,
    3: 25
}

df["health_index"] = (
    df["health_code"]
    .map(health_map)
)


print(
    df[
        [
            "health_state",
            "health_code",
            "health_index"
        ]
    ].head(10)
)


# ============================================================
# 16. SAVE MODEL PACKAGE
# ============================================================

model_package = {

    # Random Forest
    "health_model": model,

    # Isolation Forest
    "anomaly_model": anomaly_model,

    # Required scaler for anomaly prediction
    "anomaly_scaler": scaler,

    # Feature list
    "features": features,

    # Health class names
    "class_names": class_names,

    # Health score mapping
    "health_map": health_map,

    # Best anomaly contamination
    "best_contamination": best_contamination,

    # Model performance information
    "health_model_accuracy": accuracy,

    "anomaly_accuracy": final_accuracy,

    "anomaly_precision": final_precision,

    "anomaly_recall": final_recall,

    "anomaly_f1": final_f1
}


joblib.dump(
    model_package,
    "industrial_motor_ai_models.joblib"
)


print("\n" + "=" * 60)
print("MODELS SAVED")
print("=" * 60)

print(
    "industrial_motor_ai_models.joblib"
)

print(
    f"Best anomaly contamination saved: "
    f"{best_contamination:.3f}"
)


# ============================================================
# 17. TEST ONE MOTOR SAMPLE
# ============================================================

sample = X_test.iloc[[0]]


# ------------------------------------------------------------
# HEALTH PREDICTION
# ------------------------------------------------------------

prediction = model.predict(
    sample
)[0]

probability = model.predict_proba(
    sample
)[0]

confidence = max(
    probability
)

health_index = health_map[
    prediction
]


# ------------------------------------------------------------
# SCALE SAMPLE FOR ISOLATION FOREST
# ------------------------------------------------------------
sample_scaled = scaler.transform(
    sample
)

sample_scaled = pd.DataFrame(
    sample_scaled,
    columns=features,
    index=sample.index
)
# ------------------------------------------------------------
# ANOMALY PREDICTION
# ------------------------------------------------------------

anomaly_result = anomaly_model.predict(
    sample_scaled
)[0]

is_anomaly = (
    anomaly_result == -1
)


# ============================================================
# 18. DISPLAY SINGLE MOTOR RESULT
# ============================================================

print("\n" + "=" * 60)
print("SINGLE MOTOR ANALYSIS")
print("=" * 60)

print("\nInput:")

print(sample)


print("\nHealth Prediction:")

print(
    class_names[prediction]
)

print(
    f"Health Index : "
    f"{health_index}%"
)

print(
    f"Confidence   : "
    f"{confidence * 100:.2f}%"
)


print("\nPrediction Probabilities:")

for i, probability_value in enumerate(
    probability
):

    print(
        f"{class_names[i]:10} : "
        f"{probability_value * 100:.2f}%"
    )


print("\nAnomaly Detection:")

if is_anomaly:

    print(
        "WARNING - Abnormal sensor pattern detected"
    )

else:

    print(
        "Normal sensor pattern"
    )


print("\n" + "=" * 60)
print("PROGRAM COMPLETED")
print("=" * 60)