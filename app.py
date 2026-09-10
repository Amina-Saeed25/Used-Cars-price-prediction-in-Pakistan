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

# Hide the +/- stepper buttons on number inputs, and dim the placeholder option
st.markdown("""
<style>
button[data-testid="stNumberInputStepUp"], button[data-testid="stNumberInputStepDown"] {
    display: none;
}
div[data-baseweb="select"] span {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Pakistani Used Car Price Predictor")

st.header("Enter Car Details")

PLACEHOLDER = "-- Select --"

def dynamic_field(label, options_series, widget_type="select"):
    """
    Shows a widget only if there is more than one real option.
    If there's exactly one option, auto-selects it and just displays it.
    If there are no options yet (Brand/Model not chosen), shows an empty/placeholder widget.
    """
    options = sorted(options_series.unique().tolist()) if options_series is not None else []

    if len(options) == 1:
        st.write(f"**{label}:** {options[0]}  *(only option available)*")
        return options[0]
    elif len(options) == 0:
        if widget_type == "select":
            st.selectbox(label, [PLACEHOLDER], disabled=True)
        else:
            st.radio(label, [], index=None)
        return None
    else:
        if widget_type == "select":
            return st.selectbox(label, [PLACEHOLDER] + options)
        else:
            return st.radio(label, options, index=None)

# --- Cascading Dropdown: Brand -> Model ---
brands = [PLACEHOLDER] + sorted(df["Brand"].unique())
selected_brand = st.selectbox("Select Brand", brands)

if selected_brand != PLACEHOLDER:
    available_models = [PLACEHOLDER] + sorted(df[df["Brand"] == selected_brand]["Model"].unique())
else:
    available_models = [PLACEHOLDER]
selected_model = st.selectbox("Select Model", available_models)

if selected_brand != PLACEHOLDER and selected_model != PLACEHOLDER:
    model_subset = df[(df["Brand"] == selected_brand) & (df["Model"] == selected_model)]
else:
    model_subset = pd.DataFrame()

# --- Year ---
selected_year = dynamic_field("Manufacture Year", model_subset["Year"] if not model_subset.empty else None, "select")
if selected_year == PLACEHOLDER:
    selected_year = None

# --- KM Driven ---
km_driven = st.number_input("KM Driven", min_value=0, max_value=1000000, value=None, step=1000, placeholder="Enter KM driven")

# --- Engine_CC ---
selected_cc = dynamic_field("Engine CC", model_subset["Engine_CC"] if not model_subset.empty else None, "select")
if selected_cc == PLACEHOLDER:
    selected_cc = None

# --- City (sorted by frequency) ---
city_counts = df["City"].value_counts()
cities = [PLACEHOLDER] + city_counts.index.tolist()
selected_city = st.selectbox("Select City", cities)

# --- Fuel Type (radio, auto-select if only one option) ---
selected_fuel = dynamic_field("Fuel Type", model_subset["Fuel_Type"] if not model_subset.empty else None, "radio")

# --- Transmission (radio, auto-select if only one option) ---
selected_transmission = dynamic_field("Transmission", model_subset["Transmission"] if not model_subset.empty else None, "radio")

# --- Body Type (radio, auto-select if only one option) ---
selected_body = dynamic_field("Body Type", model_subset["Body_Type"] if not model_subset.empty else None, "radio")

# --- Assembly (radio, auto-select if only one option) ---
selected_assembly = dynamic_field("Assembly", model_subset["Assembly"] if not model_subset.empty else None, "radio")

# --- Color ---
colors = [PLACEHOLDER] + sorted(df["Color"].unique())
selected_color = st.selectbox("Color", colors)

# --- Owner Type (radio, full list) ---
owner_options = sorted(df["Owner_Type"].unique())
selected_owner = st.radio("Owner Type", owner_options, index=None)

# --- Registered In (sorted by frequency, like City) ---
registered_counts = df["Registered_In"].value_counts()
registered_locations = [PLACEHOLDER] + registered_counts.index.tolist()
selected_registered = st.selectbox("Registered In", registered_locations)


st.header("Predicted Price")

if st.button("Predict Price"):

    missing_fields = []
    if selected_brand == PLACEHOLDER:
        missing_fields.append("Brand")
    if selected_model == PLACEHOLDER:
        missing_fields.append("Model")
    if selected_year is None:
        missing_fields.append("Manufacture Year")
    if km_driven is None:
        missing_fields.append("KM Driven")
    if selected_cc is None:
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

        input_data = {
            "Engine_CC": selected_cc,
            "Car_Age": car_age,
            "Brand_Freq": df[df["Brand"] == selected_brand].shape[0],
            "Model_Freq": model_subset.shape[0],
            "City_Freq": df[df["City"] == selected_city].shape[0],
            "Registered_In_Freq": df[df["Registered_In"] == selected_registered].shape[0],
        }

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

        km_bins = [0, 25000, 50000, 75000, 100000, 150000, 200000, float("inf")]
        km_labels = ["0-25k", "25k-50k", "50k-75k", "75k-100k", "100k-150k", "150k-200k", "200k+"]
        km_range_label = pd.cut([km_driven], bins=km_bins, labels=km_labels)[0]
        km_range_col = f"KM_Range_{km_range_label}"
        if km_range_col in input_data:
            input_data[km_range_col] = 1

        input_df = pd.DataFrame([input_data])[model_columns]

        tree_preds_log = [tree.predict(input_df.values)[0] for tree in model.estimators_]
        tree_preds_actual = np.expm1(tree_preds_log)

        point_estimate = np.expm1(model.predict(input_df.values)[0])
        lower = np.percentile(tree_preds_actual, 5) * 0.92
        upper = np.percentile(tree_preds_actual, 95) * 1.08

        st.success(f"Estimated Price: PKR {point_estimate:,.0f}")
        st.info(f"Price Range: PKR {lower:,.0f} — PKR {upper:,.0f}")