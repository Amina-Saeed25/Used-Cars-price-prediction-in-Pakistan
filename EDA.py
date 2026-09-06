import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the cleaned dataset
df = pd.read_csv(r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\all_cars_data_cleaned.csv")

# Create a folder to save all EDA plots
output_dir = r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\eda_plots"
os.makedirs(output_dir, exist_ok=True)

# Set a consistent style for all plots
sns.set_style("whitegrid")
'''
# --- Price_PKR distribution: histogram ---
plt.figure(figsize=(10, 6))
sns.histplot(df["Price_PKR"], bins=50, kde=True)
plt.title("Price Distribution (PKR)")
plt.xlabel("Price (PKR)")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, "price_distribution.png"))
plt.show()

# --- Price_PKR distribution: boxplot (to visualize spread and outliers) ---
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["Price_PKR"])
plt.title("Price Boxplot (PKR)")
plt.xlabel("Price (PKR)")
plt.savefig(os.path.join(output_dir, "price_boxplot.png"))
plt.show()

# --- Log-transformed Price_PKR distribution (to check if it looks more normal) ---
import numpy as np
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(df["Price_PKR"]), bins=50, kde=True)
plt.title("Log-Transformed Price Distribution")
plt.xlabel("Log(Price + 1)")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, "price_log_distribution.png"))
plt.show()

print("Plots saved to:", output_dir)

'''



'''
# --- Year distribution ---
plt.figure(figsize=(12, 6))
sns.histplot(df["Year"], bins=40, kde=False)
plt.title("Car Year Distribution")
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, "year_distribution.png"))
plt.show()

# --- KM_Driven distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(df["KM_Driven"], bins=50, kde=True)
plt.title("KM Driven Distribution")
plt.xlabel("KM Driven")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, "km_distribution.png"))
plt.show()

# --- Brand-wise count (top 15 brands only, for readability) ---
plt.figure(figsize=(12, 8))
top_brands = df["Brand"].value_counts().head(15)
sns.barplot(x=top_brands.values, y=top_brands.index, orient="h")
plt.title("Top 15 Brands by Listing Count")
plt.xlabel("Count")
plt.ylabel("Brand")
plt.savefig(os.path.join(output_dir, "top_brands.png"))
plt.show()

print("Plots saved to:", output_dir)
'''


'''
# --- Price vs Year scatter plot ---
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="Year", y="Price_PKR", alpha=0.2)
plt.title("Price vs Year")
plt.xlabel("Year")
plt.ylabel("Price (PKR)")
plt.savefig(os.path.join(output_dir, "price_vs_year.png"))
plt.show()

# --- Price vs KM_Driven scatter plot ---
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="KM_Driven", y="Price_PKR", alpha=0.2)
plt.title("Price vs KM Driven")
plt.xlabel("KM Driven")
plt.ylabel("Price (PKR)")
plt.savefig(os.path.join(output_dir, "price_vs_km.png"))
plt.show()

print("Plots saved to:", output_dir)
'''



# --- Price by Fuel_Type ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Fuel_Type", y="Price_PKR")
plt.title("Price by Fuel Type")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "price_by_fuel.png"))
plt.show()

# --- Price by Transmission ---
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Transmission", y="Price_PKR")
plt.title("Price by Transmission")
plt.savefig(os.path.join(output_dir, "price_by_transmission.png"))
plt.show()

# --- Price by Assembly (Local vs Imported) ---
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Assembly", y="Price_PKR")
plt.title("Price by Assembly")
plt.savefig(os.path.join(output_dir, "price_by_assembly.png"))
plt.show()

# --- Price by Body_Type ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Body_Type", y="Price_PKR")
plt.title("Price by Body Type")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "price_by_bodytype.png"))
plt.show()

print("Plots saved to:", output_dir)