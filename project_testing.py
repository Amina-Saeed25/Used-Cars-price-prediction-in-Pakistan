import pandas as pd

# Load the cleaned dataset
df = pd.read_csv(r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\all_cars_data_cleaned.csv")

# Filter for Corolla, Year between 2015-2017, KM_Driven between 50000-80000
result = df[
    (df["Model"] == "Corolla") &
    (df["Year"].between(2015, 2017)) &
    (df["KM_Driven"].between(50000, 80000))
]

print("Number of matching rows:", len(result))
print(result[["Year", "KM_Driven", "Price_PKR"]])