import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("car_price_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

@st.cache_data
def load_data():
    return pd.read_csv("all_cars_data_cleaned.csv")

model, model_columns = load_model()
df = load_data()

st.markdown("""
<style>
button[data-testid="stNumberInputStepUp"], button[data-testid="stNumberInputStepDown"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# --- Custom tire icon (original illustration, embedded as inline SVG) ---
tire_svg = """
<svg width="70" height="70" viewBox="0 0 400 400" role="img" style="vertical-align:middle;">
<defs>
<radialGradient id="tireGrad" cx="40%" cy="35%" r="70%">
<stop offset="0%" stop-color="#3a3a3d"/>
<stop offset="70%" stop-color="#1a1a1c"/>
<stop offset="100%" stop-color="#0a0a0b"/>
</radialGradient>
<radialGradient id="rimGrad" cx="35%" cy="30%" r="70%">
<stop offset="0%" stop-color="#e8e9eb"/>
<stop offset="45%" stop-color="#a6a9ae"/>
<stop offset="100%" stop-color="#6b6e73"/>
</radialGradient>
<radialGradient id="hubGrad" cx="40%" cy="35%" r="70%">
<stop offset="0%" stop-color="#d4d6d9"/>
<stop offset="100%" stop-color="#8a8d92"/>
</radialGradient>
</defs>
<circle cx="200" cy="200" r="150" fill="url(#tireGrad)"/>
<g stroke="#050506" stroke-width="3" opacity="0.85">
<line x1="200" y1="50" x2="200" y2="72"/>
<line x1="326" y1="88" x2="313" y2="105"/>
<line x1="200" y1="350" x2="200" y2="328"/>
<line x1="74" y1="88" x2="87" y2="105"/>
<line x1="326" y1="312" x2="313" y2="295"/>
<line x1="74" y1="312" x2="87" y2="295"/>
<line x1="350" y1="200" x2="328" y2="200"/>
<line x1="50" y1="200" x2="72" y2="200"/>
<line x1="316" y1="140" x2="298" y2="152"/>
<line x1="84" y1="140" x2="102" y2="152"/>
<line x1="316" y1="260" x2="298" y2="248"/>
<line x1="84" y1="260" x2="102" y2="248"/>
</g>
<circle cx="200" cy="200" r="150" fill="none" stroke="#050506" stroke-width="4"/>
<circle cx="200" cy="200" r="112" fill="url(#rimGrad)" stroke="#4d4f53" stroke-width="1.5"/>
<g fill="#7a7d82" stroke="#4d4f53" stroke-width="1">
<path d="M200 200 L200 96 L212 98 L214 198 Z"/>
<path d="M200 200 L288 148 L296 158 L210 202 Z"/>
<path d="M200 200 L288 252 L280 264 L198 206 Z"/>
<path d="M200 200 L200 304 L188 302 L186 202 Z"/>
<path d="M200 200 L112 252 L104 242 L190 198 Z"/>
<path d="M200 200 L112 148 L120 136 L202 196 Z"/>
</g>
<circle cx="200" cy="200" r="34" fill="url(#hubGrad)" stroke="#4d4f53" stroke-width="1.5"/>
<circle cx="200" cy="200" r="9" fill="#3a3c40"/>
<g fill="#3a3c40">
<circle cx="200" cy="176" r="4"/>
<circle cx="221" cy="188" r="4"/>
<circle cx="221" cy="212" r="4"/>
<circle cx="200" cy="224" r="4"/>
<circle cx="179" cy="212" r="4"/>
<circle cx="179" cy="188" r="4"/>
</g>
</svg>
"""

st.markdown(f'<div style="display:flex; align-items:center; gap:12px;">{tire_svg}<h1 style="margin:0;">Pakistani Used Car Price Predictor</h1></div>', unsafe_allow_html=True)

st.header("Enter Car Details")

PLACEHOLDER = "-- Select --"

# --- Step 1: Brand -> Model (always active) ---
brands = [PLACEHOLDER] + sorted(df["Brand"].unique())
selected_brand = st.selectbox("Select Brand", brands)

if selected_brand != PLACEHOLDER:
    available_models = [PLACEHOLDER] + sorted(df[df["Brand"] == selected_brand]["Model"].unique())
else:
    available_models = [PLACEHOLDER]
selected_model = st.selectbox("Select Model", available_models)

# Everything below stays locked until Brand AND Model are both chosen
car_chosen = (selected_brand != PLACEHOLDER) and (selected_model != PLACEHOLDER)

if car_chosen:
    model_subset = df[(df["Brand"] == selected_brand) & (df["Model"] == selected_model)]
else:
    model_subset = pd.DataFrame()

def dynamic_field(label, options_series, widget_type, locked):
    if locked:
        if widget_type == "select":
            st.selectbox(label, [PLACEHOLDER], disabled=True)
        else:
            st.radio(label, [], index=None, disabled=True)
        return None

    options = sorted(options_series.unique().tolist())

    if len(options) == 1:
        st.write(f"**{label}:** {options[0]}  *(only option available)*")
        return options[0]
    else:
        if widget_type == "select":
            result = st.selectbox(label, [PLACEHOLDER] + options)
            return None if result == PLACEHOLDER else result
        else:
            return st.radio(label, options, index=None)

# --- Step 2: Everything else, unlocked together once car_chosen is True ---
selected_year = dynamic_field("Manufacture Year", model_subset["Year"] if car_chosen else None, "select", locked=not car_chosen)

km_driven = st.number_input(
    "KM Driven", min_value=0, max_value=1000000, value=None, step=1000,
    placeholder="Enter KM driven", disabled=not car_chosen
)

selected_cc = dynamic_field("Engine CC", model_subset["Engine_CC"] if car_chosen else None, "select", locked=not car_chosen)

city_counts = df["City"].value_counts()
cities = [PLACEHOLDER] + city_counts.index.tolist()
selected_city_raw = st.selectbox("Select City", cities, disabled=not car_chosen)
selected_city = None if selected_city_raw == PLACEHOLDER else selected_city_raw

selected_fuel = dynamic_field("Fuel Type", model_subset["Fuel_Type"] if car_chosen else None, "radio", locked=not car_chosen)
selected_transmission = dynamic_field("Transmission", model_subset["Transmission"] if car_chosen else None, "radio", locked=not car_chosen)
selected_body = dynamic_field("Body Type", model_subset["Body_Type"] if car_chosen else None, "radio", locked=not car_chosen)
selected_assembly = dynamic_field("Assembly", model_subset["Assembly"] if car_chosen else None, "radio", locked=not car_chosen)

colors = [PLACEHOLDER] + sorted(df["Color"].unique())
selected_color_raw = st.selectbox("Color", colors, disabled=not car_chosen)
selected_color = None if selected_color_raw == PLACEHOLDER else selected_color_raw

owner_options = sorted(df["Owner_Type"].unique())
selected_owner = st.radio("Owner Type", owner_options, index=None, disabled=not car_chosen)

registered_counts = df["Registered_In"].value_counts()
registered_locations = [PLACEHOLDER] + registered_counts.index.tolist()
selected_registered_raw = st.selectbox("Registered In", registered_locations, disabled=not car_chosen)
selected_registered = None if selected_registered_raw == PLACEHOLDER else selected_registered_raw


st.header("Predicted Price")

if st.button("Predict Price", disabled=not car_chosen):

    missing_fields = []
    if selected_year is None:
        missing_fields.append("Manufacture Year")
    if km_driven is None:
        missing_fields.append("KM Driven")
    if selected_cc is None:
        missing_fields.append("Engine CC")
    if selected_city is None:
        missing_fields.append("City")
    if selected_fuel is None:
        missing_fields.append("Fuel Type")
    if selected_transmission is None:
        missing_fields.append("Transmission")
    if selected_body is None:
        missing_fields.append("Body Type")
    if selected_assembly is None:
        missing_fields.append("Assembly")
    if selected_color is None:
        missing_fields.append("Color")
    if selected_owner is None:
        missing_fields.append("Owner Type")
    if selected_registered is None:
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