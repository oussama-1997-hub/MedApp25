import streamlit as st
st.set_page_config(page_title="Technique Survival Predictor", layout="wide")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Technique Survival Predictor", layout="wide")

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

with st.expander("📊 View a Sample of the Dataset", expanded=False):
    st.markdown("**First 5 rows from the dataset:**")
    st.markdown('<div class="df-container">', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
  -----------------------------------------------------------------------------------

target = 'technique_survival_levels'

# 3) Identify columns
# — Binary columns are those whose unique values ⊆ {0,1}, excluding the target
binary_cols = [c for c in df.columns 
               if set(df[c].dropna().unique()) <= {0,1} and c != target]

# — Multi-category (non-numeric) columns
multi_cat_cols = ['scholarship level ', 'Initial_nephropathy',
                  'Technique', 'Permeability_type', 'Germ']

# — Special 2‑value categorical columns (stored as 1/2 in your data)
gender_col = 'Gender '
origin_col = 'Rural_or_Urban_Origin'
gender_map = {"Male":1, "Female":2}
origin_map = {"Urban":2, "Rural":1}

# 4) Encode for model training
df_enc = df.copy()

# a) Map gender & origin (already numeric 1/2 in df)
#    so we leave df_enc[gender_col] and df_enc[origin_col] as is

# b) Label‑encode our multi-category columns
le_dict = {}
for col in multi_cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

# Everything else (binary, numeric) stays numeric

# 5) Prepare X/y and scale
X = df_enc.drop(columns=[target])
y = df_enc[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a simple Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, y)

# 6) Streamlit UI
st.title("🎯 Technique Survival Level Predictor")
st.markdown(
    "Fill in the patient’s data below. "
    "✔️ Use dropdowns and checkboxes to select the option that matches your patient."
)

with st.form("patient_form"):
    # — Demographics
    with st.expander("👤 Demographics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=int(df['Age'].mean()))
        with c2:
            gender = st.selectbox("Gender", ["Male","Female"], help="Select Male or Female")
        c1, c2 = st.columns(2)
        with c1:
            origin = st.selectbox("Residence", ["Urban","Rural"], help="Select Urban or Rural origin")
        with c2:
            transpl = st.checkbox("Transplant before Dialysis",
                                  help="Check if patient had kidney transplant before starting PD")

    # — Socioeconomic
    with st.expander("💼 Socioeconomic Status", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            schol = st.selectbox("Scholarship Level",
                                 df['scholarship level '].dropna().unique().tolist(),
                                 help="Educational level")
        with c2:
            indig = st.checkbox("Indigent CNAM Coverage",
                                help="Check if patient is under indigent CNAM scheme")

    # — Medical History (binary)
    with st.expander("🩺 Medical History", expanded=False):
        c1, c2 = st.columns(2)
        for i, col in enumerate(binary_cols):
            with (c1 if i%2==0 else c2):
                val = st.checkbox(col.replace("_"," ").title(), help=f"Check if patient has {col.replace('_',' ').lower()}")
                locals()[col] = int(val)

    # — Dialysis Parameters
    with st.expander("💧 Dialysis Parameters", expanded=False):
        # Numeric inputs
        numeric_list = [c for c in df.columns 
                        if c not in binary_cols + multi_cat_cols 
                        + [gender_col, origin_col, target]]
        c1, c2 = st.columns(2)
        for i, col in enumerate(numeric_list):
            with (c1 if i%2==0 else c2):
                locals()[col] = st.number_input(col.replace("_"," ").title(),
                                                value=float(df[col].mean()),
                                                help=f"Enter patient’s {col}")

        # Multi-category inputs
        c1, c2 = st.columns(2)
        for i, col in enumerate(multi_cat_cols):
            with (c1 if i%2==0 else c2):
                locals()[col] = st.selectbox(
                    col.strip(),
                    df[col].dropna().unique().tolist(),
                    help=f"Select patient’s {col.strip().lower()}"
                )

    submitted = st.form_submit_button("🔍 Predict")

# 7) On submit, assemble and predict
if submitted:
    inp = {}

    # Demographics
    inp['Age'] = age
    inp[gender_col] = gender_map[gender]
    inp[origin_col] = origin_map[origin]
    inp['transplant_before_dialysis'] = int(transpl)

    # Socioeconomic
    inp['scholarship level '] = schol
    inp['Indigent_Coverage_CNAM'] = int(indig)

    # Medical History
    for col in binary_cols:
        inp[col] = locals()[col]

    # Dialysis numeric & multi-cat
    for col in numeric_list:
        inp[col] = locals()[col]
    for col in multi_cat_cols:
        inp[col] = locals()[col]

    # Build DataFrame
    input_df = pd.DataFrame([inp])

    # Label‑encode multi-cat
    for col in multi_cat_cols:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))

    # Scale & predict
    input_scaled = scaler.transform(input_df[X.columns])
    pred = clf.predict(input_scaled)[0]
    st.success(f"🎉 Predicted Technique Survival Level: **{pred}**")
