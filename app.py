import streamlit as st
st.set_page_config(page_title="Technique Survival Predictor", layout="wide")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Streamlit app title
st.title("Prediction of Target Using Decision Tree 2222")
st.write("Please fill in the details below to predict the target.")

# Load the data
url = "https://raw.githubusercontent.com/oussama-1997-hub/MedApp25/main/BD%20sans%20encod%20stand.xlsx"
df = pd.read_excel(url, engine="openpyxl")

# Display the data in the app (optional)
st.write("Here is the dataset:")
st.dataframe(df.head())

st.title("Technique Survival Level Prediction")
st.markdown("Enter patient information below to predict the **technique survival level**.")

# =======================
# Preprocessing
# =======================
# Define target and categorical columns
# Define columns
target = 'technique_survival_levels'
categorical_cols = ['Gender ', 'Rural_or_Urban_Origin', 'scholarship level ', 'Indigent_Coverage_CNAM',
                    'Smoking', 'Diabetes', 'Hypertension', 'Heart_Disease', 'Gout', 'Cancer',
                    'Pneumothorax', 'Psychiatric_disorder', 'Initial_nephropathy',
                    'Technique', 'Permeability_type', 'Diuretic', 'ACEI_ARB', 'Icodextrin', 'Autonomy']

# === Encoding ===
df_encoded = df.copy()
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# === Prepare data ===
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_scaled, y)

# Load the trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🎯 Technique Survival Level Predictor")

st.markdown("Provide patient information below to predict technique survival level.")

# Categorical options mapping
gender_map = {"Male": 1, "Female": 2}
urban_map = {"Urban": 2, "Rural": 1}
transplant_map = {True: 1, False: 0}

# Define the list of input fields (simplified demo — expand as needed)
categorical_fields = {
    "Gender": list(gender_map.keys()),
    "Rural_or_Urban_Origin": list(urban_map.keys()),
}

binary_fields = [
    "transplant_before_dialysis", "Diabetes", "Hypertension", "Heart_Disease",
    "Gout", "Cancer", "Pneumothorax", "Psychiatric_disorder",
    "Diuretic", "ACEI_ARB", "Icodextrin", "Autonomy"
]

numerical_fields = [
    "Age", "BMI", "Hemoglobin", "Albumin", "Residual_diuresis", "Creatinine_clearance"
]

# Create input form
with st.form("prediction_form"):
    st.subheader("📋 Patient Input")

    col1, col2 = st.columns(2)

    inputs = {}

    # Handle categorical features
    with col1:
        inputs["Gender"] = gender_map[st.selectbox("Gender", categorical_fields["Gender"])]
        inputs["Rural_or_Urban_Origin"] = urban_map[st.selectbox("Urban or Rural Origin", categorical_fields["Rural_or_Urban_Origin"])]
    
    # Handle binary features
    for i, field in enumerate(binary_fields):
        col = col1 if i % 2 == 0 else col2
        inputs[field] = transplant_map[col.checkbox(field.replace("_", " ").title())]

    # Handle numerical features
    for i, field in enumerate(numerical_fields):
        col = col1 if i % 2 == 0 else col2
        inputs[field] = col.number_input(field.replace("_", " ").title(), step=0.1)

    # Submit button
    submitted = st.form_submit_button("🔍 Predict")

# Make prediction
if submitted:
    input_df = pd.DataFrame([inputs])

    # Scale the input using the pre-fitted scaler
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.success(f"🎉 Predicted Technique Survival Level: **{prediction}**")
