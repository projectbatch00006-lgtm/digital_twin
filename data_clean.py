# ============================================
# INDUSTRIAL MOTOR DATA - DATA CLEANING PHASE
# ============================================

import pandas as pd
import numpy as np

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------

file_path = "finalised_industrial_motor_random_forest_dataset.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------
# 2. Check Missing Values
# --------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print("\nTotal Missing Values:", total_missing)

if total_missing == 0:
    print("✓ No missing values found.")
else:
    print("⚠ Missing values found.")


# --------------------------------------------
# 3. Check Duplicate Rows
# --------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

if duplicate_count == 0:
    print("✓ No duplicate rows found.")
else:
    print("⚠ Duplicate rows found.")

    # Display duplicates
    print("\nDuplicate rows:")
    print(df[df.duplicated()])


# --------------------------------------------
# 4. Remove Exact Duplicate Rows
# --------------------------------------------

if duplicate_count > 0:

    df = df.drop_duplicates().reset_index(drop=True)

    print("\n✓ Duplicate rows removed.")

else:

    print("\n✓ No duplicate removal required.")


# --------------------------------------------
# 5. Define Numerical Features
# --------------------------------------------

numeric_features = [
    "voltage_v",
    "current_a",
    "temperature_c",
    "vibration",
    "power_w",
    "thermal_load"
]

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

print(numeric_features)


# --------------------------------------------
# 6. Outlier Detection using IQR
# --------------------------------------------

print("\n" + "=" * 60)
print("OUTLIER CHECK - IQR METHOD")
print("=" * 60)

outlier_summary = []

for column in numeric_features:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    outlier_count = len(outliers)

    outlier_summary.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outliers": outlier_count
    })

    print(f"\nFeature: {column}")
    print(f"Q1           : {Q1:.4f}")
    print(f"Q3           : {Q3:.4f}")
    print(f"IQR          : {IQR:.4f}")
    print(f"Lower Bound  : {lower_bound:.4f}")
    print(f"Upper Bound  : {upper_bound:.4f}")
    print(f"Outliers     : {outlier_count}")


# --------------------------------------------
# 7. Outlier Summary Table
# --------------------------------------------

outlier_df = pd.DataFrame(outlier_summary)

print("\n" + "=" * 60)
print("OUTLIER SUMMARY")
print("=" * 60)

print(outlier_df.to_string(index=False))


# --------------------------------------------
# 8. Create Outlier Flag
# --------------------------------------------

df["outlier_flag"] = False

for column in numeric_features:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    condition = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    df.loc[condition, "outlier_flag"] = True


# --------------------------------------------
# 9. Count Rows Having At Least One Outlier
# --------------------------------------------

total_outlier_rows = df["outlier_flag"].sum()

print("\n" + "=" * 60)
print("TOTAL OUTLIER ROWS")
print("=" * 60)

print("Rows containing at least one outlier:",
      total_outlier_rows)


# --------------------------------------------
# 10. Important Decision
# --------------------------------------------

print("\n" + "=" * 60)
print("OUTLIER DECISION")
print("=" * 60)

print("""
Outliers have NOT been automatically deleted.

Reason:
Extreme motor measurements can represent genuine
abnormal operating conditions and may be useful for
motor-health classification.

They will therefore be flagged for inspection first.
""")


# --------------------------------------------
# 11. Dataset Summary After Cleaning
# --------------------------------------------

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print("Final Rows    :", len(df))
print("Final Columns :", len(df.columns))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# --------------------------------------------
# 12. Save Cleaned Dataset
# --------------------------------------------

output_file = "industrial_motor_cleaned.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("CLEANED DATASET SAVED")
print("=" * 60)

print(f"File: {output_file}")