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
categorical_cols = ['Gender ', 'Rural_or_Urban_Origin', 'scholarship level ', 'Indigent_Coverage_CNAM',
                    'Smoking', 'Diabetes', 'Hypertension', 'Heart_Disease', 'Gout', 'Cancer',
                    'Pneumothorax', 'Psychiatric_disorder', 'Initial_nephropathy',
                    'Technique', 'Permeability_type', 'Diuretic', 'ACEI_ARB', 'Icodextrin', 'Autonomy']

df_encoded = df.copy()
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    le_dict[col] = le

# Split features and target
X = df_encoded.drop(columns=['technique_survival_levels'])
y = df_encoded['technique_survival_levels']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# =======================
# Streamlit Form
# =======================
st.sidebar.header("Input Features")

def user_input_features():
    input_data = {}
    for col in categorical_cols:
        options = df[col].dropna().unique()
        selected = st.sidebar.selectbox(col.strip(), sorted(options))
        input_data[col] = selected
    return input_data

input_dict = user_input_features()

# =======================
if st.button("Predict"):
    # Create input DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Encode using stored LabelEncoders
    for col in categorical_cols:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))
    
    # Reorder columns to match training data
    input_df = input_df[X.columns]  # <- This line ensures same order and names
    
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = clf.predict(input_scaled)[0]
    st.success(f"Predicted Technique Survival Level: {prediction}")
