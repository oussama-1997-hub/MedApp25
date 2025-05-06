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
    /* Title */
    .big-title {
        font-size: 3rem;
        font-weight: 700;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    /* Subtitle */
    .subtitle {
        font-size: 1.2rem;
        color: #566573;
        text-align: center;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    /* Container around the dataframe */
    .df-container {
        background-color: #F2F4F4;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── HEADER & INSTRUCTIONS ─────────────────────────────────────────────────────
st.markdown('<div class="big-title">Technique Survival Level Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Enter your patient’s details below. '
    'Once complete, click **Predict** to see their technique survival level.'
    '</div>',
    unsafe_allow_html=True
)

# ─── LOAD & DISPLAY DATASET SAMPLE ──────────────────────────────────────────────
@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/"
        "oussama-1997-hub/MedApp25/main/"
        "BD%20sans%20encod%20stand.xlsx"
    )
    return pd.read_excel(url, engine="openpyxl")

df = load_data()

with st.expander("📊 View Sample Data", expanded=False):
    st.markdown("**First 5 rows of the dataset:**")
    st.markdown('<div class="df-container">', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── 4) PREPARE MODEL DATA ───────────────────────────────────────────────────────
target = 'technique_survival_levels'
drop_feats = ['BMI_one_year', 'RRF_one_year', 'Technique_survival']

df_model = df.drop(columns=drop_feats)

binary_cols = [c for c in df_model.columns 
               if set(df_model[c].dropna().unique()) <= {0,1} and c != target]

multi_cat_cols = ['scholarship level ', 'Initial_nephropathy',
                  'Technique', 'Permeability_type', 'Germ']

gender_col = 'Gender '
origin_col = 'Rural_or_Urban_Origin'
gender_map = {"Male":1, "Female":2}
origin_map = {"Urban":2, "Rural":1}

# Encode
from sklearn.preprocessing import LabelEncoder
le_dict = {}
df_enc = df_model.copy()
for col in multi_cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

# Features & scaling
X = df_enc.drop(columns=[target])
y = df_enc[target]
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = DecisionTreeClassifier(random_state=42).fit(X_scaled, y)

# ─── 5) INPUT FORM ───────────────────────────────────────────────────────────────
st.markdown("### 📝 Patient Data Input")
form = st.form("patient_form")
# Demographics
with form.expander("👤 Demographics", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=int(df['Age'].mean()))
    with c2:
        gender = st.selectbox("Gender", list(gender_map.keys()))
    c1, c2 = st.columns(2)
    with c1:
        origin = st.selectbox("Residence", list(origin_map.keys()))
    with c2:
        transpl = st.checkbox("Transplant before Dialysis")

# Socioeconomic
with form.expander("💼 Socioeconomic Status", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        schol_opts = sorted(df['scholarship level '].unique().tolist())
        schol = st.selectbox("Scholarship Level", schol_opts)
    with c2:
        indig = st.checkbox("Indigent CNAM Coverage")

# Medical History
with form.expander("🩺 Medical History", expanded=False):
    c1, c2 = st.columns(2)
    for i, col in enumerate(binary_cols):
        with (c1 if i%2==0 else c2):
            val = st.checkbox(col.replace("_"," ").title())
            locals()[col] = int(val)

# Dialysis Parameters
with form.expander("💧 Dialysis Parameters", expanded=False):
    dialysis_nums = ['BMI_start_PD','Urine_output_start','Initial_RRF',
                     'Initial_UF','Initial_albumin','Initial_Hb','Nbre_peritonitis']
    d1, d2 = st.columns(2)
    for i, col in enumerate(dialysis_nums):
        with (d1 if i%2==0 else d2):
            locals()[col] = st.number_input(col.replace("_"," ").title(), value=float(df[col].mean()))
    d1, d2 = st.columns(2)
    pt_opts = sorted(df['Permeability_type'].unique().tolist())
    germ_opts = sorted(df['Germ'].unique().tolist())
    with d1:
        locals()['Permeability_type'] = st.selectbox("Permeability Type", pt_opts)
    with d2:
        locals()['Germ'] = st.selectbox("Germ", germ_opts)

# Submit button inside form
submitted = form.form_submit_button("🔍 Predict")

# ─── 6) MAKE PREDICTION ──────────────────────────────────────────────────────────
if submitted:
    inp = {
        'Age': age,
        gender_col: gender_map[gender],
        origin_col: origin_map[origin],
        'transplant_before_dialysis': int(transpl),
        'scholarship level ': schol,
        'Indigent_Coverage_CNAM': int(indig)
    }
    for col in binary_cols:
        inp[col] = locals()[col]
    for col in dialysis_nums:
        inp[col] = locals()[col]
    inp['Permeability_type'] = locals()['Permeability_type']
    inp['Germ'] = locals()['Germ']

    input_df = pd.DataFrame([inp])
    for col in multi_cat_cols:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))

    input_scaled = scaler.transform(input_df[X.columns])
    pred = clf.predict(input_scaled)[0]

    if pred == 2:
        st.success("✅ **Will succeed ≥ 2 years** (Level 2)")
        st.info("This PD technique is expected to succeed for at least two years.")
    else:
        st.error(f"⚠️ **Not expected to exceed 2 years** (Level {pred})")
        st.warning("Consider closer monitoring or alternative strategies for long-term success.")


