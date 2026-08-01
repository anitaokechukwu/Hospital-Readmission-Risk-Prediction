# 🏥 Hospital Readmission Risk Prediction Using Machine Learning

> An end-to-end Machine Learning project that predicts whether a patient is likely to be readmitted to the hospital within 30 days using demographic, clinical, and hospitalization data.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-black?style=for-the-badge&logo=pandas)

---

# 📌 Project Overview

Hospital readmissions are a major challenge for healthcare systems because they increase healthcare costs, reduce hospital efficiency, and may indicate gaps in patient care.

This project uses **Machine Learning** to predict whether a patient will be readmitted within **30 days after discharge** based on clinical and demographic information.

The application is deployed as an interactive **Streamlit web application**, allowing users to enter patient information and instantly receive a readmission risk prediction.

---

# 🚀 Live Demo

🔗 **Streamlit App**

> *(Add your Streamlit deployment link here after deployment.)*

Example:

https://your-app-name.streamlit.app

---

# 📷 Application Preview

## Dashboard

![Dashboard](images/dashboard.png)

---

## Prediction Result

![Prediction](images/prediction.png)

---

## Feature Importance

![Feature Importance](images/feature_importance.png)

---

# 🎯 Business Problem

Hospital readmissions place a significant financial burden on healthcare providers and may indicate poor continuity of care.

Early identification of high-risk patients allows healthcare professionals to:

- Improve discharge planning
- Schedule earlier follow-up appointments
- Reduce avoidable readmissions
- Improve patient outcomes
- Optimize healthcare resources

---

# 📂 Dataset

**Source:** Kaggle

Dataset contains approximately **18,000 patient records** with demographic, clinical, and hospitalization information.

### Features include:

- Age
- Gender
- Insurance Type
- Length of Stay
- Severity Score
- Previous Admissions
- Previous Readmissions
- HbA1c Level
- Creatinine Level
- Hemoglobin Level
- Medication Count
- Medication Adherence Score
- Blood Pressure
- ICU Stay
- Comorbidity Index
- Chronic Disease Count
- Follow-up Appointment
- Discharge Disposition
- Admission Type

**Target Variable**

- Readmitted Within 30 Days
  - 0 = No
  - 1 = Yes

> **Note:** This dataset appears to be synthetic and is intended for educational and portfolio purposes.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Streamlit
- Jupyter Notebook

---

# 📊 Exploratory Data Analysis

EDA included:

- Missing value analysis
- Target variable visualization
- Feature distributions
- Correlation analysis
- Outlier detection
- Readmission analysis by age
- Gender distribution
- Admission type analysis

---

# ⚙ Data Preprocessing

The following preprocessing steps were performed:

- Handling missing values
- One-Hot Encoding
- Feature Scaling
- Train/Test Split
- Feature Selection

---

# 🤖 Machine Learning Models

The following classification algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Random Forest was selected as the final model because it provided the most reliable overall performance for this dataset.

---

# 📈 Model Evaluation

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Cross Validation

---

# ⭐ Feature Importance

The Random Forest model identified the following features as the most influential:

- Time Since Last Discharge
- Age
- Average Systolic Blood Pressure
- Length of Stay
- Number of Medications
- Hemoglobin Level
- Socioeconomic Risk Score
- Severity Score
- HbA1c Level
- Comorbidity Index

---

# 💻 Streamlit Application

The interactive application allows users to:

✅ Enter patient information

✅ Predict readmission risk

✅ View prediction probability

✅ View patient summary

✅ Interpret clinical risk level

✅ Explore feature importance

---

# 📁 Project Structure

```text
Hospital-Readmission-Risk-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── Hospital Readmission Risk Prediction.ipynb
├── hospital_readmission_risk_dataset_2026_v1_18000rows.csv
│
├── models/
│   ├── random_forest_readmission_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── images/
    ├── dashboard.png
    ├── prediction.png
    └── feature_importance.png
```

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Hospital-Readmission-Risk-Prediction.git
```

Move into the project folder:

```bash
cd Hospital-Readmission-Risk-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

- Deep Learning implementation
- Explainable AI (SHAP)
- Hyperparameter tuning
- Cloud deployment enhancements
- Integration with Electronic Health Records (EHR)

---

# 👩‍💻 Author

**Anita Okechukwu**

Healthcare Data Analyst | Machine Learning Enthusiast | Registered Midwife

I am passionate about applying Data Analytics and Machine Learning to solve healthcare challenges and improve patient outcomes.

- GitHub: https://github.com/YOUR_USERNAME
- LinkedIn: https://linkedin.com/in/YOUR_LINKEDIN

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps others discover my work and supports my learning journey.
