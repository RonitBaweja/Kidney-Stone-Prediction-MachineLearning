import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIGURE
# ---------------------------------------------------

st.set_page_config(
    page_title="Kidney Stone Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

artifacts = joblib.load("kidney_stone_model.pkl")

model = artifacts["model"]

scaler = artifacts["scaler"]

feature_names = artifacts["feature_names"]

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🩺 Kidney Stone Prediction System")

st.markdown("""
This system predicts the likelihood of kidney stone occurrence using 
machine learning models trained on blood and urine diagnostic parameters.
""")

st.divider()

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30,
        help="Ideal adult range: 18–65 years"
    )

with col2:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

st.divider()

st.subheader("🧪 Clinical Parameters")

col1, col2 = st.columns(2)

with col1:

    urine_calcium = st.number_input(
        "Urine Calcium (mg/day)",
        value=180.0,
        help="Ideal range: 100–250 mg/day"
    )

    urine_oxalate = st.number_input(
        "Urine Oxalate (mg/day)",
        value=30.0,
        help="Ideal range: 20–40 mg/day"
    )

    urine_volume = st.number_input(
        "Urine Volume (L/day)",
        value=2.5,
        help="Ideal range: 2–3 L/day"
    )

    serum_uric_acid = st.number_input(
        "Serum Uric Acid (mg/dL)",
        value=5.0,
        help="Ideal range: 3.5–7.2 mg/dL"
    )

with col2:

    serum_potassium = st.number_input(
        "Serum Potassium (mEq/L)",
        value=4.2,
        help="Ideal range: 3.5–5.0 mEq/L"
    )

    water_intake = st.number_input(
        "Daily Water Intake (L)",
        value=2.5,
        help="Ideal range: 2–4 L/day"
    )

    activity = st.selectbox(
        "Physical Activity Level",
        ["Low", "Moderate", "High"]
    )

    previous_stone = st.selectbox(
        "Previous Kidney Stone",
        ["No", "Yes"]
    )

st.divider()

# ---------------------------------------------------
# ENCODING INPUTS
# ---------------------------------------------------

gender_encoded = 1 if gender == "Male" else 0

activity_map = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

activity_encoded = activity_map[activity]

previous_stone_encoded = 1 if previous_stone == "Yes" else 0

# ---------------------------------------------------
# CREATE INPUT DATA
# ---------------------------------------------------

sample_data = {

    "Age": age,
    "Gender": gender_encoded,
    "Urine_Calcium_mg_day": urine_calcium,
    "Urine_Oxalate_mg_day": urine_oxalate,
    "Urine_Volume_L_day": urine_volume,
    "Serum_UricAcid_mg_dL": serum_uric_acid,
    "Serum_Potassium_mEq_L": serum_potassium,
    "Daily_Water_Intake_L": water_intake,
    "Physical_Activity_Level": activity_encoded,
    "Previous_Kidney_Stone": previous_stone_encoded
}

# Fill remaining features
for feature in feature_names:

    if feature not in sample_data:

        sample_data[feature] = 0

# Convert to DataFrame
sample_df = pd.DataFrame([sample_data])

# Correct column order
sample_df = sample_df[feature_names]

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict Kidney Stone Risk"):

    # Scale data
    sample_scaled = scaler.transform(sample_df)

    # Prediction
    prediction = model.predict(sample_scaled)[0]

    probability = model.predict_proba(sample_scaled)[0][1]

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Kidney Stone Detected")

    else:

        st.success("✅ No Kidney Stone Detected")

    st.write(f"### Prediction Probability: {probability:.2%}")

    # Risk level
    if probability < 0.35:

        st.success("🟢 Risk Level: LOW")

    elif probability < 0.70:

        st.warning("🟡 Risk Level: MODERATE")

    else:

        st.error("🔴 Risk Level: HIGH")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption("Machine Learning Based Kidney Stone Prediction System\n")
st.caption("Priyamvad Ranjan 2K22/EP/076")
st.caption("Ritik Kumar 2K22/EP/085")
st.caption("Ronit Baweja 2K22/EP/086")