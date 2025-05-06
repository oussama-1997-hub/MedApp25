import streamlit as st
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
target = 'technique_survival_levels'
categorical_cols = ['Gender ', 'Rural_or_Urban_Origin', 'scholarship level ', 'Indigent_Coverage_CNAM',
                    'Smoking', 'Diabetes', 'Hypertension', 'Heart_Disease', 'Gout', 'Cancer',
                    'Pneumothorax', 'Psychiatric_disorder', 'Initial_nephropathy',
                    'Technique', 'Permeability_type', 'Diuretic', 'ACEI_ARB', 'Icodextrin', 'Autonomy']

# Encode data for training
df_encoded = df.copy()
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# Separate features and target
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, y)

# ============ Streamlit App ============

st.title("Technique Survival Prediction")

st.markdown("Provide patient and clinical details to predict the technique survival level.")

# Collect user input
input_data = {}
for col in X.columns:
    if col in categorical_cols:
        unique_vals = df[col].dropna().unique().tolist()
        input_val = st.selectbox(f"{col}", unique_vals)
        input_data[col] = input_val
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_val = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
        input_data[col] = input_val

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical features
for col in categorical_cols:
    input_df[col] = le_dict[col].transform(input_df[col])

# Align column order
input_df = input_df[X.columns]

# Scale
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Technique Survival Level"):
    prediction = clf.predict(input_scaled)[0]
    st.success(f"Predicted Technique Survival Level: {prediction}")
