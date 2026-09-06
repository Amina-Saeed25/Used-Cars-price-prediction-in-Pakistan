import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the fully encoded, model-ready dataset
df = pd.read_csv(r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\all_cars_data_encoded.csv")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

# Drop raw/text columns and the target's raw source (avoid data leakage)
drop_cols = ["Brand", "Model", "Year", "Price_PKR", "KM_Driven", "City", "Registered_In", "Ad_URL", "Price_log"]

X = df.drop(columns=drop_cols)
y = df["Price_log"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeature columns:\n", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# --- Train and compare models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)

    # Convert predictions back from log scale to actual PKR for interpretable metrics
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"{name}: RMSE={rmse:,.0f} PKR | MAE={mae:,.0f} PKR | R2={r2:.4f}")

results_df = pd.DataFrame(results)
print("\n--- Comparison Table ---")
print(results_df)

# --- Price Range Prediction using Random Forest tree spread ---

# Get predictions from every individual tree in the forest (not just the averaged final prediction)
rf_model = models["Random Forest"]
tree_predictions_log = np.array([tree.predict(X_test) for tree in rf_model.estimators_])
# Shape: (n_trees, n_test_samples)

# Convert each tree's prediction back to actual PKR
tree_predictions_actual = np.expm1(tree_predictions_log)

# Calculate a range using 5th and 95th percentile across trees (90% confidence range)
lower_bound = np.percentile(tree_predictions_actual, 5, axis=0)
upper_bound = np.percentile(tree_predictions_actual, 95, axis=0)
point_estimate = np.expm1(rf_model.predict(X_test))
actual_price = np.expm1(y_test.values)


# Add a safety margin to widen the range further (helps cover cases where tree spread underestimates true uncertainty)
margin = 0.08  # 8% buffer
lower_bound = lower_bound * (1 - margin)
upper_bound = upper_bound * (1 + margin)

within_range = ((actual_price >= lower_bound) & (actual_price <= upper_bound)).mean()
print(f"\n% of actual prices falling within predicted range (with margin): {within_range * 100:.1f}%")


# Show first 10 examples: actual vs predicted range
comparison = pd.DataFrame({
    "Actual_Price": actual_price[:10],
    "Predicted_Point": point_estimate[:10],
    "Range_Lower": lower_bound[:10],
    "Range_Upper": upper_bound[:10]
})
print(comparison.round(0))

# Check how often the actual price falls within the predicted range
within_range = ((actual_price >= lower_bound) & (actual_price <= upper_bound)).mean()
print(f"\n% of actual prices falling within predicted range: {within_range * 100:.1f}%")



import joblib

# Save the trained Random Forest model (best performer) for later use in the Streamlit app
joblib.dump(rf_model, r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\car_price_model.pkl")

# Save the feature column order too -- important! The app must send data in the exact same column order
joblib.dump(list(X.columns), r"D:\PycharmProjects\SummerProject\Car-Price-Prediction\model_columns.pkl")

print("Model and column list saved successfully.")