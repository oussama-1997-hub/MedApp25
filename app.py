import streamlit as st
st.set_page_config(page_title="Technique Survival Predictor", layout="wide")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── STYLING ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .big-title {
        font-size: 3rem;
        font-weight: 700;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #566573;
        text-align: center;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .df-container {
        background-color: #F2F4F4;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .highlight {
        background-color: #EBF5FB;
        padding: 1rem;
        border: 2px solid #3498DB;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="big-title">Technique Survival Level Predictor 222</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Enter your patient’s details below. '
    'Once complete, click **Predict** to see their technique survival level.'
    '</div>',
    unsafe_allow_html=True
)

# ─── LOAD DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/oussama-1997-hub/MedApp25/main/BD%20sans%20encod%20stand.xlsx"
    return pd.read_excel(url, engine="openpyxl")

df = load_data()

with st.expander("📊 View Sample Data", expanded=False):
    st.markdown("**First 5 rows of the dataset:**")
    st.markdown('<div class="df-container">', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── MODEL SETUP ────────────────────────────────────────────────────────────────
target = 'technique_survival_levels'
drop_feats = ['BMI_one_year', 'RRF_one_year', 'Technique_survival']
df_model = df.drop(columns=drop_feats)

binary_cols = [c for c in df_model.columns if set(df_model[c].dropna().unique()) <= {0,1} and c != target]
multi_cat_cols = ['scholarship level ', 'Initial_nephropathy', 'Technique', 'Permeability_type', 'Germ']
gender_col = 'Gender '
origin_col = 'Rural_or_Urban_Origin'
gender_map = {"Male":1, "Female":2}
origin_map = {"Urban":2, "Rural":1}

# Encoding
df_enc = df_model.copy()
le_dict = {}
for col in multi_cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

X = df_enc.drop(columns=[target])
y = df_enc[target]
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = DecisionTreeClassifier(random_state=42).fit(X_scaled, y)

# ─── TOP FEATURES ───────────────────────────────────────────────────────────────
top_features = [
    'Age', 'BMI_start_PD', 'Initial_RRF ', 'Initial_albumin',
    'Nbre_peritonitis', 'Germ', 'scholarship level ',
    'Hypertension', 'Initial_Charlson_score', 'Autonomy'
]

# ─── INPUT FORM ─────────────────────────────────────────────────────────────────
st.markdown("### 🌟 Key Features (Required)")
with st.form("patient_form"):
    st.markdown(
        '<div class="highlight">'
        'Please fill in the **most important** features below for accurate predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    key_inputs = {}

    # Row 1
    c1, c2 = st.columns(2)
    key_inputs['Age'] = c1.number_input(
        "Age (years)",
        min_value=0, max_value=120,
        value=int(df['Age'].mean())
    )
    key_inputs['BMI_start_PD'] = c2.number_input(
        "BMI at Start of PD",
        value=float(df['BMI_start_PD'].mean())
    )

    # Row 2
    c1, c2 = st.columns(2)
    key_inputs['Initial_RRF '] = c1.number_input(
        "Initial RRF",
        value=float(df['Initial_RRF '].mean())
    )
    key_inputs['Initial_albumin'] = c2.number_input(
        "Initial Albumin",
        value=float(df['Initial_albumin'].mean())
    )

    # Row 3
    c1, c2 = st.columns(2)
    key_inputs['Nbre_peritonitis'] = c1.number_input(
        "Number of Peritonitis Episodes",
        min_value=0,
        value=int(df['Nbre_peritonitis'].mean())
    )
    key_inputs['Germ'] = c2.number_input(
        "Germ (encoded)",
        value=float(df['Germ'].mean())
    )

    # Row 4
    c1, c2 = st.columns(2)
    key_inputs['scholarship level '] = c1.number_input(
        "Scholarship Level (encoded)",
        value=float(df['scholarship level '].mean())
    )
    key_inputs['Hypertension'] = c2.number_input(
        "Hypertension (0=no, 1=yes)",
        min_value=0, max_value=1,
        value=int(df['Hypertension'].mean())
    )

    # Row 5
    c1, c2 = st.columns(2)
    key_inputs['Initial_Charlson_score'] = c1.number_input(
        "Initial Charlson Score",
        min_value=0,
        value=int(df['Initial_Charlson_score'].mean())
    )
    key_inputs['Autonomy'] = c2.number_input(
        "Autonomy (0=no, 1=yes)",
        min_value=0, max_value=1,
        value=int(df['Autonomy'].mean())
    )

    
    # ─── OPTIONAL SECTIONS ──────────────────────────────────────────────────────
    st.markdown("### 🧩 Optional Inputs (for more precision)")

    with st.expander("👤 Demographics", expanded=False):
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", list(gender_map.keys()))
        origin = c2.selectbox("Residence", list(origin_map.keys()))
        transpl = c1.checkbox("Transplant before Dialysis")

    with st.expander("💼 Socioeconomic Status", expanded=False):
        c1, c2 = st.columns(2)
        schol = c1.selectbox("Scholarship Level", df['scholarship level '].dropna().unique().tolist())
        indig = c2.checkbox("Indigent CNAM Coverage")

    with st.expander("🩺 Medical History", expanded=False):
        for i, col in enumerate(binary_cols):
            col_title = col.replace("_"," ").title()
            val = st.checkbox(col_title)
            key_inputs[col] = int(val)

    with st.expander("💧 Dialysis Parameters", expanded=False):
        numeric_list = [c for c in df.columns if c not in binary_cols + multi_cat_cols 
                        + ['Gender ', 'Rural_or_Urban_Origin', 'transplant_before_dialysis', target]
                        + drop_feats]
        for col in numeric_list:
            if col not in top_features:
                key_inputs[col] = st.number_input(col.replace("_"," ").title(), value=float(df[col].mean()))
        for col in multi_cat_cols:
            options = sorted(df[col].dropna().unique().tolist())
            default_val = options[0] if options else None
            key_inputs[col] = st.selectbox(col.strip().replace("_", " ").title(), options, 
                                           index=options.index(default_val) if default_val in options else 0)

    submitted = st.form_submit_button("🔍 Predict")

# ─── PREDICTION ─────────────────────────────────────────────────────────────────
if submitted:
    inp = dict(key_inputs)  # Start with top features and filled optional ones
    inp['Gender '] = gender_map[gender]
    inp['Rural_or_Urban_Origin'] = origin_map[origin]
    inp['transplant_before_dialysis'] = int(transpl)
    inp['scholarship level '] = schol
    inp['Indigent_Coverage_CNAM'] = int(indig)

    input_df = pd.DataFrame([inp])

    for col in multi_cat_cols:
        if col in input_df.columns:
            input_df[col] = le_dict[col].transform(input_df[col].astype(str))

    input_scaled = scaler.transform(input_df[X.columns])
    pred = clf.predict(input_scaled)[0]

    if pred == 2:
        st.success("✅ Predicted Technique Survival Level: 2 (will succeed ≥ 2 years)")
        st.info("This PD technique is expected to succeed for at least **2 years**, indicating a good prognosis.")
    else:
        st.error(f"⚠️ Predicted Technique Survival Level: {pred} (will not exceed 2 years)")
        st.warning("This PD technique may not last beyond **2 years**; consider close monitoring or alternative strategies.")
