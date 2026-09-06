import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\all_cars_data_cleaned.csv")
print("Loaded shape:", df.shape)

# 1. Car_Age feature
current_year = 2026
df["Car_Age"] = current_year - df["Year"]

# 2. KM_Range feature
km_bins = [0, 25000, 50000, 75000, 100000, 150000, 200000, float("inf")]
km_labels = ["0-25k", "25k-50k", "50k-75k", "75k-100k", "100k-150k", "150k-200k", "200k+"]
df["KM_Range"] = pd.cut(df["KM_Driven"], bins=km_bins, labels=km_labels)

# 3. Price_log feature
df["Price_log"] = np.log1p(df["Price_PKR"])

print(df[["Year", "Car_Age", "KM_Driven", "KM_Range", "Price_PKR", "Price_log"]].head(10))
print("\nCar_Age stats:")
print(df["Car_Age"].describe())
print("\nKM_Range counts:")
print(df["KM_Range"].value_counts())

df.to_csv(r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\all_cars_data_features.csv", index=False)
print("\nSaved feature-engineered file.")

# --- 4. Frequency Encoding for high-cardinality columns ---
for col in ["Brand", "Model", "City", "Registered_In"]:
    freq_map = df[col].value_counts()
    df[col + "_Freq"] = df[col].map(freq_map)

df["Color"] = df["Color"].str.strip().str.capitalize()

# --- 5. One-Hot Encoding for genuinely low-cardinality columns only ---
onehot_cols = ["Fuel_Type", "Transmission", "Body_Type",
               "Assembly", "Color", "Owner_Type", "KM_Range"]

print("Unique value counts for one-hot columns:")
for col in onehot_cols:
    print(f"  {col}: {df[col].nunique()} unique values")

df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

print("\nShape before encoding:", df.shape)
print("Shape after encoding:", df_encoded.shape)

df_encoded.to_csv(r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\all_cars_data_encoded.csv", index=False)
print("\nSaved encoded file.")