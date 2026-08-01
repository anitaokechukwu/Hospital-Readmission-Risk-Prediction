import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Hospital Readmission Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("models/random_forest_readmission_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# =====================================================
# TITLE
# =====================================================

st.title("🏥 Hospital Readmission Risk Prediction")

st.markdown("""
Predict whether a patient is likely to be **readmitted within 30 days**
using a Machine Learning Random Forest model.

This application was built using:

- Python
- Scikit-Learn
- Streamlit
- Random Forest Classifier
""")

st.info(
    "This application is for educational purposes only and should not be used for clinical decision-making."
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Patient Information")

# -----------------------------
# Demographics
# -----------------------------

st.sidebar.subheader("Demographics")

age = st.sidebar.slider(
    "Age",
    18,
    100,
    50
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

insurance = st.sidebar.selectbox(
    "Insurance Type",
    [
        "Other",
        "Medicare",
        "Private"
    ]
)

# -----------------------------
# Admission
# -----------------------------

st.sidebar.subheader("Admission")

admission_type = st.sidebar.selectbox(
    "Admission Type",
    [
        "Elective",
        "Emergency",
        "Urgent"
    ]
)

diagnosis = st.sidebar.selectbox(
    "Primary Diagnosis",
    [
        "Cardiac",
        "Diabetes",
        "Infection",
        "Other",
        "Respiratory"
    ]
)

discharge = st.sidebar.selectbox(
    "Discharge Disposition",
    [
        "Home",
        "Nursing Facility",
        "Rehab"
    ]
)

# =====================================================
# CLINICAL INFORMATION
# =====================================================

st.header("Clinical Information")

col1, col2 = st.columns(2)

with col1:

    socioeconomic = st.slider(
        "Socioeconomic Risk Score",
        1,
        10,
        5
    )

    previous_admissions = st.number_input(
        "Previous Admissions (6 Months)",
        0,
        20,
        1
    )

    previous_readmissions = st.number_input(
        "Previous Readmissions (1 Year)",
        0,
        10,
        0
    )

    time_since_discharge = st.slider(
        "Days Since Last Discharge",
        0,
        365,
        30
    )

    length_of_stay = st.slider(
        "Length of Stay",
        1,
        30,
        5
    )

    comorbidity = st.slider(
        "Comorbidity Index",
        0,
        10,
        2
    )

    chronic = st.slider(
        "Chronic Disease Count",
        0,
        10,
        2
    )

    icu = st.selectbox(
        "ICU Stay",
        ["No", "Yes"]
    )

with col2:

    severity = st.slider(
        "Severity Score",
        1,
        10,
        5
    )

    hba1c = st.number_input(
        "HbA1c Level",
        4.0,
        15.0,
        6.5
    )

    creatinine = st.number_input(
        "Creatinine Level",
        0.3,
        10.0,
        1.1
    )

    hemoglobin = st.number_input(
        "Hemoglobin Level",
        5.0,
        20.0,
        13.5
    )

    systolic_bp = st.slider(
        "Average Systolic BP",
        80,
        220,
        120
    )

    medications = st.slider(
        "Number of Medications",
        0,
        20,
        5
    )

    medication_changes = st.slider(
        "Medication Change Count",
        0,
        10,
        1
    )

    high_risk_med = st.selectbox(
        "High Risk Medication",
        ["No", "Yes"]
    )

followup = st.selectbox(
    "Follow-up Appointment Scheduled",
    ["No", "Yes"]
)

adherence = st.slider(
    "Medication Adherence Score",
    0.0,
    100.0,
    75.0
)

st.divider()

predict_button = st.button(
    "Predict Readmission Risk",
    use_container_width=True
)


# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    # Create a dictionary with all features initialized to 0
    patient = {feature: 0 for feature in feature_names}

    # -----------------------------
    # Numerical Features
    # -----------------------------

    patient["Age"] = age
    patient["Socioeconomic_Risk_Score"] = socioeconomic
    patient["Previous_Admissions_6M"] = previous_admissions
    patient["Previous_Readmissions_1Y"] = previous_readmissions
    patient["Time_Since_Last_Discharge"] = time_since_discharge
    patient["Length_of_Stay"] = length_of_stay
    patient["Comorbidity_Index"] = comorbidity
    patient["Chronic_Disease_Count"] = chronic
    patient["ICU_Stay_Flag"] = 1 if icu == "Yes" else 0
    patient["Severity_Score"] = severity
    patient["HbA1c_Level"] = hba1c
    patient["Creatinine_Level"] = creatinine
    patient["Hemoglobin_Level"] = hemoglobin
    patient["Average_Systolic_BP"] = systolic_bp
    patient["Number_of_Medications"] = medications
    patient["Medication_Change_Count"] = medication_changes
    patient["High_Risk_Medication_Flag"] = 1 if high_risk_med == "Yes" else 0
    patient["Followup_Appointment_Scheduled"] = 1 if followup == "Yes" else 0
    patient["Medication_Adherence_Score"] = adherence

    # -----------------------------
    # One-Hot Encoded Features
    # -----------------------------

    if gender == "Male":
        patient["Gender_Male"] = 1

    if insurance == "Medicare":
        patient["Insurance_Type_Medicare"] = 1
    elif insurance == "Private":
        patient["Insurance_Type_Private"] = 1

    if admission_type == "Emergency":
        patient["Admission_Type_Emergency"] = 1
    elif admission_type == "Urgent":
        patient["Admission_Type_Urgent"] = 1

    if diagnosis == "Diabetes":
        patient["Primary_Diagnosis_Group_Diabetes"] = 1
    elif diagnosis == "Infection":
        patient["Primary_Diagnosis_Group_Infection"] = 1
    elif diagnosis == "Other":
        patient["Primary_Diagnosis_Group_Other"] = 1
    elif diagnosis == "Respiratory":
        patient["Primary_Diagnosis_Group_Respiratory"] = 1

    if discharge == "Nursing Facility":
        patient["Discharge_Disposition_Nursing Facility"] = 1
    elif discharge == "Rehab":
        patient["Discharge_Disposition_Rehab"] = 1

    # -----------------------------
    # Create DataFrame
    # -----------------------------

    input_df = pd.DataFrame([patient])

    # Ensure correct feature order
    input_df = input_df[feature_names]

    # Scale the data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.divider()

    st.header("Prediction Results")

    if prediction == 1:

        st.error("🔴 High Risk of Readmission")

    else:

        st.success("🟢 Low Risk of Readmission")

    st.metric(
        "Probability of Readmission",
        f"{probability*100:.2f}%"
    )


    st.subheader("Patient Summary")
    
    summary = pd.DataFrame({
    "Feature": [
        "Age",
        "Length of Stay",
        "Severity Score",
        "Previous Admissions",
        "Medication Count"
    ],
    "Value": [
        age,
        length_of_stay,
        severity,
        previous_admissions,
        medications
    ]
    })

    st.dataframe(summary, use_container_width=True)


    st.subheader("Clinical Interpretation")

    if probability >= 0.80:

        st.error("""
High predicted readmission risk.

Consider:

• Close post-discharge monitoring

• Medication review

• Early follow-up appointment

• Patient education
""")

    elif probability >= 0.50:

        st.warning("""
Moderate predicted readmission risk.

Additional monitoring may reduce the chance of readmission.
""")

    else:

        st.success("""
Low predicted readmission risk.

Continue routine discharge planning and follow-up.
""")




    st.divider()

st.header("About This Model")

st.markdown("""
### Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Dataset Size:** 18,000 patient records
- **Input Features:** 30
- **Prediction Target:** Readmission Within 30 Days

### Technologies Used

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Joblib

### Project Purpose

This project demonstrates how machine learning can be applied to predict hospital readmission risk using patient demographic and clinical information.

The application is intended for educational and portfolio purposes.
""")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.divider()

st.header("📊 Top 15 Most Important Features")

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(15)
)

st.bar_chart(
    importance_df.set_index("Feature")
)

st.dataframe(
    importance_df,
    use_container_width=True
)

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.divider()

st.header("📈 Model Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", "Random Forest")
col2.metric("Dataset Size", "18,000")
col3.metric("Features", "30")

st.markdown("""
### Model Highlights

- Random Forest Classifier
- 18,000 synthetic patient records
- 30 engineered features
- Predicts 30-day hospital readmission risk

### Important Note

The dataset used for this project appears to be synthetic and is intended for educational and portfolio purposes.
Predictions should **not** be used for real clinical decision-making.
""")

st.divider()

st.markdown("---")

st.markdown(
"""
### 👩‍💻 Developed by Anita Okechukwu

**Healthcare Data Analytics Portfolio Project**

**Tools Used**

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Joblib

© 2026 Anita Okechukwu
"""
)











