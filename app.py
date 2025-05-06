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

# ========== ENCODING ==========
categorical_cols = ['Gender ', 'Rural_or_Urban_Origin', 'scholarship level ', 'Indigent_Coverage_CNAM',
                    'Smoking', 'Diabetes', 'Hypertension', 'Heart_Disease', 'Gout', 'Cancer',
                    'Pneumothorax', 'Psychiatric_disorder', 'Initial_nephropathy',
                    'Technique', 'Permeability_type', 'Diuretic', 'ACEI_ARB', 'Icodextrin', 'Autonomy']

le_dict = {}
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# ========== CORRELATION MATRIX ==========
corr = df_encoded.corr(numeric_only=True)

# ========== FEATURES AND TARGET ==========
X = df_encoded.drop(columns=['technique_survival_levels'])
y = df_encoded['technique_survival_levels']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# ========== TRAIN DECISION TREE ==========
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# ========== STREAMLIT APP ==========
st.title("🎯 Technique Survival Predictor")

st.markdown("Provide the patient information below to predict the technique survival level.")

input_data = {}

# === CATEGORICAL FIELDS (Decoded) ===
gender = st.selectbox("Gender", ['Male', 'Female'])
input_data['Gender '] = gender

rural_urban = st.selectbox("Residence", ['Urban', 'Rural'])
input_data['Rural_or_Urban_Origin'] = rural_urban

transplant = st.checkbox("Transplanted Before Dialysis")
input_data['transplant_before_dialysis'] = 1 if transplant else 0

# === BOOLEAN FIELDS ===
bool_cols = ['Smoking', 'Diabetes', 'Hypertension', 'Heart_Disease', 'Gout',
             'Cancer', 'Pneumothorax', 'Psychiatric_disorder', 'Diuretic',
             'ACEI_ARB', 'Icodextrin', 'Autonomy', 'Indigent_Coverage_CNAM']

col1, col2, col3 = st.columns(3)
for i, col in enumerate(bool_cols):
    with [col1, col2, col3][i % 3]:
        input_data[col] = 1 if st.checkbox(col.replace("_", " ")) else 0

# === REMAINING CATEGORICAL FIELDS ===
cat_cols = ['scholarship level ', 'Initial_nephropathy', 'Technique', 'Permeability_type']
for col in cat_cols:
    options = df_original[col].unique().tolist()
    selected = st.selectbox(col.strip(), options)
    input_data[col] = selected

# === NUMERICAL FIELDS ===
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['technique_survival_levels']]
for col in numerical_cols:
    input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

# ========== PREDICTION ==========
if st.button("🔍 Predict Technique Survival Level"):
    # Encode the inputs
    input_df = pd.DataFrame([input_data])

    for col in categorical_cols:
        if col in input_df.columns:
            le = le_dict[col]
            input_df[col] = le.transform(input_df[col])

    # Scale input
    input_scaled = scaler.transform(input_df[X.columns])

    prediction = clf.predict(input_scaled)[0]
    st.success(f"🎯 Predicted Technique Survival Level: **{prediction}**")

    # Optional: Show class probabilities
    probs = clf.predict_proba(input_scaled)[0]
    st.write("Prediction Probabilities:")
    st.bar_chart(pd.Series(probs, name="Probability"))

# ========== OPTIONAL: CONFUSION MATRIX ==========
with st.expander("Show Confusion Matrix and Model Evaluation"):
    st.subheader("Confusion Matrix")
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
    st.pyplot(fig)
