import pandas as pd
import numpy as np

# Always load from the ORIGINAL merged file to avoid compounding errors
df = pd.read_csv(r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\all_cars_data_merged.csv")
print("Start shape:", df.shape)

# 1. Drop rows missing the target variable
df = df.dropna(subset=["Price_PKR"])

# 2. Body_Type -> fill using Brand+Model mode, fallback "Unknown"
df["Body_Type"] = df.groupby(["Brand", "Model"])["Body_Type"].transform(
    lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
)
df["Body_Type"] = df["Body_Type"].fillna("Unknown")

# 3. Engine_CC -> fix invalid values (0 or >6000) then fill missing, both via Brand+Model median
df.loc[(df["Engine_CC"] == 0) | (df["Engine_CC"] > 6000), "Engine_CC"] = None
df["Engine_CC"] = df.groupby(["Brand", "Model"])["Engine_CC"].transform(lambda x: x.fillna(x.median()))
df["Engine_CC"] = df["Engine_CC"].fillna(df["Engine_CC"].median())

# 4. Fuel_Type, Registered_In, Assembly, Color -> fill with "Unknown"
for col in ["Fuel_Type", "Registered_In", "Assembly", "Color"]:
    df[col] = df[col].fillna("Unknown")

# 5. KM_Driven -> fix placeholder value (exactly 1,000,000) via Brand+Model+Year median
df.loc[df["KM_Driven"] == 1000000, "KM_Driven"] = None
df["KM_Driven"] = df.groupby(["Brand", "Model", "Year"])["KM_Driven"].transform(lambda x: x.fillna(x.median()))
df["KM_Driven"] = df["KM_Driven"].fillna(df["KM_Driven"].median())

print("End shape:", df.shape)
print("\nRemaining missing values:\n", df.isnull().sum())

df.to_csv(r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\all_cars_data_cleaned.csv", index=False)
print("\nSaved fully cleaned file.")