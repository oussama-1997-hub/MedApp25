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

# === Streamlit UI ===
st.title("🔬 Technique Survival Level Predictor")
st.markdown("Use the form below to input patient and clinical data. The model will predict the **technique survival level**.")

st.divider()
st.subheader("📝 Input Patient Data")

input_data = {}
num_cols = [col for col in X.columns if col not in categorical_cols]

# Display Categorical Features in 3 Columns
with st.expander("🧬 Categorical Inputs", expanded=True):
    cat_cols = st.columns(3)
    for idx, col in enumerate(categorical_cols):
        with cat_cols[idx % 3]:
            options = df[col].dropna().unique().tolist()
            selected = st.selectbox(col.strip(), options)
            input_data[col] = selected

# Display Numerical Features in 3 Columns
with st.expander("📊 Numerical Inputs", expanded=True):
    num_input_cols = st.columns(3)
    for idx, col in enumerate(num_cols):
        with num_input_cols[idx % 3]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_val = st.number_input(col.strip(), min_value=min_val, max_value=max_val, value=mean_val)
            input_data[col] = input_val

# === Prediction Button ===
st.divider()
if st.button("🚀 Predict Technique Survival Level"):
    input_df = pd.DataFrame([input_data])

    # Apply Label Encoding
    for col in categorical_cols:
        input_df[col] = le_dict[col].transform(input_df[col])

    # Align columns
    input_df = input_df[X.columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"🎯 **Predicted Technique Survival Level: {prediction}**")
