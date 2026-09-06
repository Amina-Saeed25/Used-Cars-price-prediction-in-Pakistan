import streamlit as st
import pandas as pd
import joblib

# Cache the model loading -- runs only once, not on every interaction (fixes MemoryError)
@st.cache_resource
def load_model():
    model = joblib.load("car_price_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

# Cache the dataset loading -- runs only once
@st.cache_data
def load_data():
    return pd.read_csv("all_cars_data_cleaned.csv")

model, model_columns = load_model()
df = load_data()

st.title(" Pakistani Used Car Price Predictor")
st.write("Dataset loaded:", df.shape)
st.write("Model loaded successfully. Expected features:", len(model_columns))


st.header("Enter Car Details")

# --- Cascading Dropdown: Brand -> Model ---
brands = sorted(df["Brand"].unique())
selected_brand = st.selectbox("Select Brand", brands)

# Only show models that belong to the selected brand (real data, as the teacher wanted)
available_models = sorted(df[df["Brand"] == selected_brand]["Model"].unique())
selected_model = st.selectbox("Select Model", available_models)

st.write(f"You selected: **{selected_brand} {selected_model}**")

# --- Filter dataset to just this Brand+Model -- every dropdown below uses THIS subset ---
model_subset = df[(df["Brand"] == selected_brand) & (df["Model"] == selected_model)]

# --- Year (dynamic range based on actual years this model was sold) ---
min_year = int(model_subset["Year"].min())
max_year = int(model_subset["Year"].max())
current_year = 2026

if min_year == max_year:
    selected_year = min_year
    st.write(f"Manufacture Year: {selected_year} (only year available for this model)")
else:
    selected_year = st.slider("Select Manufacture Year", min_value=min_year, max_value=max_year, value=max_year)

car_age = current_year - selected_year
st.write(f"Car Age: {car_age} years")

# --- KM Driven (dynamic max based on this model's real data) ---
max_km = int(model_subset["KM_Driven"].max())
km_driven = st.number_input("KM Driven", min_value=0, max_value=max_km, value=min(50000, max_km), step=1000)

# --- Engine_CC (dynamic: only show CC values this model actually comes in) ---
available_cc = sorted(model_subset["Engine_CC"].unique())
selected_cc = st.selectbox("Engine CC", available_cc)

# --- City (kept as full list -- city doesn't depend on car model) ---
city_counts = df["City"].value_counts()
cities = city_counts.index.tolist()  # already sorted by frequency, most common first
selected_city = st.selectbox("Select City", cities)

# --- Fuel Type (dynamic: only fuel types this model actually comes in) ---
available_fuels = sorted(model_subset["Fuel_Type"].unique())
selected_fuel = st.selectbox("Fuel Type", available_fuels)

# --- Transmission (dynamic) ---
available_transmissions = sorted(model_subset["Transmission"].unique())
selected_transmission = st.selectbox("Transmission", available_transmissions)

# --- Body Type (dynamic: this model usually has ONE body type) ---
available_bodies = sorted(model_subset["Body_Type"].unique())
selected_body = st.selectbox("Body Type", available_bodies)

# --- Assembly (dynamic) ---
available_assemblies = sorted(model_subset["Assembly"].unique())
selected_assembly = st.selectbox("Assembly", available_assemblies)

# --- Color (kept as full list -- any model can come in any color) ---
colors = sorted(df["Color"].unique())
selected_color = st.selectbox("Color", colors)

# --- Owner Type (kept as full list) ---
owner_types = sorted(df["Owner_Type"].unique())
selected_owner = st.selectbox("Owner Type", owner_types)

# --- Registered In (kept as full list) ---
registered_locations = sorted(df["Registered_In"].unique())
selected_registered = st.selectbox("Registered In", registered_locations)


st.header("Predicted Price")

if st.button("Predict Price"):
    # Build a single-row dataframe matching the training data structure
    input_data = {
        "Engine_CC": selected_cc,
        "Car_Age": car_age,
        "Brand_Freq": df[df["Brand"] == selected_brand].shape[0],
        "Model_Freq": model_subset.shape[0],
        "City_Freq": df[df["City"] == selected_city].shape[0],
        "Registered_In_Freq": df[df["Registered_In"] == selected_registered].shape[0],
    }

    # Initialize all one-hot columns to 0, then set the selected ones to 1
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    # Set the correct one-hot flags based on user selections
    fuel_col = f"Fuel_Type_{selected_fuel}"
    if fuel_col in input_data:
        input_data[fuel_col] = 1

    if selected_transmission == "Manual":
        input_data["Transmission_Manual"] = 1

    body_col = f"Body_Type_{selected_body}"
    if body_col in input_data:
        input_data[body_col] = 1

    if selected_assembly == "Local":
        input_data["Assembly_Local"] = 1
    elif selected_assembly == "Unknown":
        input_data["Assembly_Unknown"] = 1

    color_col = f"Color_{selected_color}"
    if color_col in input_data:
        input_data[color_col] = 1

    if selected_owner == "Second Owner or More":
        input_data["Owner_Type_Second Owner or More"] = 1
    elif selected_owner == "Unknown":
        input_data["Owner_Type_Unknown"] = 1

    # Determine KM_Range bin based on km_driven, matching the original binning logic
    km_bins = [0, 25000, 50000, 75000, 100000, 150000, 200000, float("inf")]
    km_labels = ["0-25k", "25k-50k", "50k-75k", "75k-100k", "100k-150k", "150k-200k", "200k+"]
    km_range_label = pd.cut([km_driven], bins=km_bins, labels=km_labels)[0]
    km_range_col = f"KM_Range_{km_range_label}"
    if km_range_col in input_data:
        input_data[km_range_col] = 1

    # Build the final input row in the EXACT column order the model expects
    input_df = pd.DataFrame([input_data])[model_columns]

    # Get predictions from all trees for a price range, plus the point estimate
    tree_preds_log = [tree.predict(input_df.values)[0] for tree in model.estimators_]
    import numpy as np
    tree_preds_actual = np.expm1(tree_preds_log)

    point_estimate = np.expm1(model.predict(input_df.values)[0])
    lower = np.percentile(tree_preds_actual, 5) * 0.92
    upper = np.percentile(tree_preds_actual, 95) * 1.08

    st.success(f"Estimated Price: PKR {point_estimate:,.0f}")
    st.info(f"Price Range: PKR {lower:,.0f} — PKR {upper:,.0f}")