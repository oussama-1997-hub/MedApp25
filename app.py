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

df_original = df.copy()

# Target column
target = 'technique_survival_levels'

# Identify binary features (0/1)
binary_cols = [col for col in df.columns if set(df[col].dropna().unique()) <= {0,1} and col != target]

# Special categorical mappings
gender_map = {1: "Male", 2: "Female"}
rural_map = {2: "Urban", 1: "Rural"}

# Define multi-category features
multi_cat_cols = ['scholarship level ', 'Initial_nephropathy', 'Technique', 'Permeability_type', 'Germ']

# Label encode all categorical for training
le_dict = {}
df_enc = df.copy()
# Encode gender and ruralurban using maps
df_enc['Gender '] = df_enc['Gender '].map({v: k for k, v in gender_map.items()})
df_enc['Rural_or_Urban_Origin'] = df_enc['Rural_or_Urban_Origin'].map({v: k for k, v in rural_map.items()})
# Encode multi-cat
for col in multi_cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le
# Encode binary (already 0/1)

# Prepare features and target
X = df_enc.drop(columns=[target])
y = df_enc[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, y)

# Streamlit UI
st.title("🎯 Technique Survival Level Predictor")
st.markdown("Provide patient data below to predict technique survival level.")

# Layout inputs in three columns
cols = st.columns(3)
input_data = {}

# 1) Gender and rural
with cols[0]:
    gender = st.selectbox("Gender", list(gender_map.values()))
    input_data['Gender '] = gender
with cols[1]:
    rural = st.selectbox("Residence", list(rural_map.values()))
    input_data['Rural_or_Urban_Origin'] = rural
with cols[2]:
    transplant = st.checkbox("Transplant Before Dialysis")
    input_data['Transplant_before_dialysis'] = int(transplant)

# 2) Binary features (checkbox)
for i, col in enumerate([c for c in binary_cols if c not in ['Transplant_before_dialysis']]):
    with cols[i % 3]:
        val = st.checkbox(col.replace('_', ' ').title())
        input_data[col] = int(val)

# 3) Multi-category features
for i, col in enumerate(multi_cat_cols):
    with cols[i % 3]:
        opts = df[col].dropna().unique().tolist()
        sel = st.selectbox(col.strip(), opts)
        input_data[col] = sel

# 4) Numerical features
num_cols = [c for c in df.columns if c not in binary_cols + multi_cat_cols + ['Gender ', 'Rural_or_Urban_Origin', 'Transplant_before_dialysis', target]]
for i, col in enumerate(num_cols):
    with cols[i % 3]:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

# Prediction
if st.button("Predict"):
    # Create DataFrame
    inp = pd.DataFrame([input_data])
    # Map back to codes
    inp['Gender '] = inp['Gender '].map({v: k for k, v in gender_map.items()})
    inp['Rural_or_Urban_Origin'] = inp['Rural_or_Urban_Origin'].map({v: k for k, v in rural_map.items()})
    for col in multi_cat_cols:
        inp[col] = le_dict[col].transform(inp[col].astype(str))
    # Scale and predict
    inp_scaled = scaler.transform(inp[X.columns])
    pred = clf.predict(inp_scaled)[0]
    st.success(f"Predicted Technique Survival Level: {pred}")

