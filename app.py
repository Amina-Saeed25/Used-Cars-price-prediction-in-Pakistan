import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cache the model loading -- runs only once, not on every interaction
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

st.title("🚗 Pakistani Used Car Price Predictor")

st.header("Enter Car Details")

PLACEHOLDER = "-- Select --"

# --- Cascading Dropdown: Brand -> Model (with placeholder so nothing is pre-selected) ---
brands = [PLACEHOLDER] + sorted(df["Brand"].unique())
selected_brand = st.selectbox("Select Brand", brands)

if selected_brand != PLACEHOLDER:
    available_models = [PLACEHOLDER] + sorted(df[df["Brand"] == selected_brand]["Model"].unique())
else:
    available_models = [PLACEHOLDER]
selected_model = st.selectbox("Select Model", available_models)

# Only filter the dataset once both Brand and Model are chosen
if selected_brand != PLACEHOLDER and selected_model != PLACEHOLDER:
    model_subset = df[(df["Brand"] == selected_brand) & (df["Model"] == selected_model)]
else:
    model_subset = pd.DataFrame()  # empty until user picks Brand+Model

# --- Year (dynamic, with placeholder) ---
if not model_subset.empty:
    year_options = [PLACEHOLDER] + sorted(model_subset["Year"].unique().tolist())
else:
    year_options = [PLACEHOLDER]
selected_year = st.selectbox("Manufacture Year", year_options)

# --- KM Driven (no default value, forces user to type) ---
km_driven = st.number_input("KM Driven", min_value=0, max_value=1000000, value=None, step=1000, placeholder="Enter KM driven")

# --- Engine_CC (dynamic, with placeholder) ---
if not model_subset.empty:
    cc_options = [PLACEHOLDER] + sorted(model_subset["Engine_CC"].unique().tolist())
else:
    cc_options = [PLACEHOLDER]
selected_cc = st.selectbox("Engine CC", cc_options)

# --- City (full list, with placeholder, sorted by frequency) ---
city_counts = df["City"].value_counts()
cities = [PLACEHOLDER] + city_counts.index.tolist()
selected_city = st.selectbox("Select City", cities)

# --- Fuel Type (radio button, dynamic to this model, nothing selected by default) ---
if not model_subset.empty:
    fuel_options = sorted(model_subset["Fuel_Type"].unique())
else:
    fuel_options = []
selected_fuel = st.radio("Fuel Type", fuel_options, index=None)

# --- Transmission (radio button, dynamic) ---
if not model_subset.empty:
    transmission_options = sorted(model_subset["Transmission"].unique())
else:
    transmission_options = []
selected_transmission = st.radio("Transmission", transmission_options, index=None)

# --- Body Type (radio button, dynamic) ---
if not model_subset.empty:
    body_options = sorted(model_subset["Body_Type"].unique())
else:
    body_options = []
selected_body = st.radio("Body Type", body_options, index=None)

# --- Assembly (radio button, dynamic) ---
if not model_subset.empty:
    assembly_options = sorted(model_subset["Assembly"].unique())
else:
    assembly_options = []
selected_assembly = st.radio("Assembly", assembly_options, index=None)

# --- Color (full list, with placeholder) ---
colors = [PLACEHOLDER] + sorted(df["Color"].unique())
selected_color = st.selectbox("Color", colors)

# --- Owner Type (radio button, full list) ---
owner_options = sorted(df["Owner_Type"].unique())
selected_owner = st.radio("Owner Type", owner_options, index=None)

# --- Registered In (full list, with placeholder) ---
registered_locations = [PLACEHOLDER] + sorted(df["Registered_In"].unique())
selected_registered = st.selectbox("Registered In", registered_locations)


st.header("Predicted Price")

if st.button("Predict Price"):

    # --- Validation: check every required field before predicting ---
    missing_fields = []
    if selected_brand == PLACEHOLDER:
        missing_fields.append("Brand")
    if selected_model == PLACEHOLDER:
        missing_fields.append("Model")
    if selected_year == PLACEHOLDER:
        missing_fields.append("Manufacture Year")
    if km_driven is None:
        missing_fields.append("KM Driven")
    if selected_cc == PLACEHOLDER:
        missing_fields.append("Engine CC")
    if selected_city == PLACEHOLDER:
        missing_fields.append("City")
    if selected_fuel is None:
        missing_fields.append("Fuel Type")
    if selected_transmission is None:
        missing_fields.append("Transmission")
    if selected_body is None:
        missing_fields.append("Body Type")
    if selected_assembly is None:
        missing_fields.append("Assembly")
    if selected_color == PLACEHOLDER:
        missing_fields.append("Color")
    if selected_owner is None:
        missing_fields.append("Owner Type")
    if selected_registered == PLACEHOLDER:
        missing_fields.append("Registered In")

    if missing_fields:
        st.error(f"⚠️ Please fill these important details before predicting: {', '.join(missing_fields)}")
    else:
        current_year = 2026
        car_age = current_year - selected_year

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

        # Determine KM_Range bin based on km_driven
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
        tree_preds_actual = np.expm1(tree_preds_log)

        point_estimate = np.expm1(model.predict(input_df.values)[0])
        lower = np.percentile(tree_preds_actual, 5) * 0.92
        upper = np.percentile(tree_preds_actual, 95) * 1.08

        st.success(f"Estimated Price: PKR {point_estimate:,.0f}")
        st.info(f"Price Range: PKR {lower:,.0f} — PKR {upper:,.0f}")