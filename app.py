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

target = 'technique_survival_levels'

# -- Identify columns --
binary_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1} and c != target]
# We’ll treat these as checkboxes (0=False, 1=True)
multi_cat_cols = ['scholarship level ', 'Initial_nephropathy',
                  'Technique', 'Permeability_type', 'Germ']
# Special two‐value categorical mappings
gender_map = {"Male": 1, "Female": 2}
origin_map = {"Urban": 2, "Rural": 1}

# -- Encode for model training --
df_enc = df.copy()
df_enc['Gender '] = df_enc['Gender '].map({v:k for k,v in gender_map.items()})
df_enc['Rural_or_Urban_Origin'] = df_enc['Rural_or_Urban_Origin'].map({v:k for k,v in origin_map.items()})

le_dict = {}
for col in multi_cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

X = df_enc.drop(columns=[target])
y = df_enc[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, y)

# 3) Title & Instructions
st.title("🎯 Technique Survival Level Predictor")
st.write(
    """
    Please fill in the following **patient details**.  
    For each field, choose or enter the option that **best matches** your patient’s profile.
    """  
)

# 4) Build the input form
with st.form("patient_form"):
    # Demographics
    with st.expander("👤 Demographics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=int(df['Age'].mean()))
        with c2:
            gender = st.selectbox("Gender", list(gender_map.keys()), help="Select patient’s gender")
        c1, c2 = st.columns(2)
        with c1:
            origin = st.selectbox("Place of Residence", list(origin_map.keys()),
                                   help="Urban or Rural origin")
        with c2:
            transplant = st.checkbox("Transplant Before Dialysis",
                                     help="Check if patient had kidney transplant before PD")

    # Socioeconomic
    with st.expander("💼 Socioeconomic Status", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            schol = st.selectbox("Scholarship Level",
                                 df['scholarship level '].dropna().unique().tolist(),
                                 help="Educational attainment level")
        with c2:
            indig = st.checkbox("Indigent CNAM Coverage",
                                help="Check if patient is covered under indigent CNAM scheme")

    # Medical History
    with st.expander("🩺 Medical History", expanded=False):
        c1, c2 = st.columns(2)
        for i, col in enumerate(binary_cols):
            with (c1 if i%2==0 else c2):
                label = col.replace("_", " ").title()
                vars()[col] = st.checkbox(label, help=f"Check if patient has {label.lower()}")

    # Dialysis Parameters
    with st.expander("💧 Dialysis Parameters", expanded=False):
        # numeric and multi-category
        all_cols = ['BMI_start_PD','BMI_one_year','Urine_output_start','Initial_RRF','RRF_one_year',
                    'Initial_UF','Initial_albumin','Initial_Hb','Nbre_peritonitis']
        c1, c2 = st.columns(2)
        for i, col in enumerate(all_cols):
            with (c1 if i%2==0 else c2):
                val = st.number_input(col.replace("_", " ").title(),
                                      value=float(df[col].mean()),
                                      help=f"Enter patient’s {col}")
        # multi-category
        c1, c2 = st.columns(2)
        for i, col in enumerate(multi_cat_cols):
            with (c1 if i%2==0 else c2):
                sel = st.selectbox(col.strip(),
                                   df[col].dropna().unique().tolist(),
                                   help=f"Select patient’s {col.strip().lower()}")

    submitted = st.form_submit_button("🔍 Predict")

# 5) Prediction logic
if submitted:
    # Gather inputs
    inp = {
        'Age': age,
        'Gender ': gender,
        'Rural_or_Urban_Origin': origin,
        'transplant_before_dialysis': int(transplant),
        'scholarship level ': schol,
        'Indigent_Coverage_CNAM': int(indig)
    }
    # Binary
    for col in binary_cols:
        inp[col] = int(locals()[col])
    # Dialysis numeric
    for col in ['BMI_start_PD','BMI_one_year','Urine_output_start','Initial_RRF','RRF_one_year',
                'Initial_UF','Initial_albumin','Initial_Hb','Nbre_peritonitis']:
        inp[col] = locals()[col]
    # Dialysis multi-cat
    for col in multi_cat_cols:
        inp[col] = locals()[col]

    # Convert to DataFrame
    input_df = pd.DataFrame([inp])

    # Map back to encoded numeric
    input_df['Gender '] = input_df['Gender '].map(gender_map)
    input_df['Rural_or_Urban_Origin'] = input_df['Rural_or_Urban_Origin'].map(origin_map)
    for col in multi_cat_cols:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))

    # Scale & predict
    input_scaled = scaler.transform(input_df[X.columns])
    pred = clf.predict(input_scaled)[0]
    st.success(f"**Predicted Technique Survival Level:** {pred}")

    # (Optional) show model confidence
    probs = clf.predict_proba(input_scaled)[0]
    st.bar_chart(pd.Series(probs, index=[f"Level {i}" for i in clf.classes_], name="Confidence"))
