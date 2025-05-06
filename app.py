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

with st.expander("📊 View a Sample of the Dataset", expanded=False):
    st.markdown("**First 5 rows from the dataset:**")
    st.markdown('<div class="df-container">', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
# ─── ML model ────────────────────────────────────────────────────────────────────

target = 'technique_survival_levels'

# ── 3) Identify columns ────────────────────────────────────────────────────────
# Binary (0/1) features
binary_cols = [
    c for c in df.columns
    if set(df[c].dropna().unique()) <= {0,1} and c != target
]

# Multi-category features
multi_cat_cols = [
    'scholarship level ', 'Initial_nephropathy',
    'Technique', 'Permeability_type', 'Germ'
]

# Two-value categorical mappings
gender_col = 'Gender '
origin_col = 'Rural_or_Urban_Origin'
gender_map = {"Male": 1, "Female": 2}
origin_map = {"Urban": 2, "Rural": 1}

# Numeric features: all except binary, multi-cat, gender/origin, and the removed ones
removed = {'BMI_one_year', 'RRF_one_year', target}
numeric_cols = [
    c for c in df.columns
    if c not in binary_cols
    and c not in multi_cat_cols
    and c not in {gender_col, origin_col}
    and c not in removed
]

# ── 4) Encode & Train Model ──────────────────────────────────────────────────
df_enc = df.copy()

# Map gender & origin (they’re already 1/2 in original)
# Label‑encode multi-category
le_dict = {}
for col in multi_cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

# Prepare X, y
X = df_enc.drop(columns=[target])
y = df_enc[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, y)

# ── 5) Page Header ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;color:#2E86C1;'>🎯 Technique Survival Predictor</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#566573;'>"
    "Fill out your patient’s data below and click “Predict” to see if the peritoneal dialysis "
    "technique is likely to succeed for at least two years."
    "</p>", unsafe_allow_html=True
)

# ── 6) Input Form ──────────────────────────────────────────────────────────────
with st.form("patient_form"):
    # Demographics
    with st.expander("👤 Demographics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(
                "Age (years)", min_value=0, max_value=120,
                value=int(df['Age'].mean()), help="Enter patient’s age"
            )
        with c2:
            gender = st.selectbox(
                "Gender", list(gender_map.keys()),
                help="Select patient’s gender"
            )
        c1, c2 = st.columns(2)
        with c1:
            origin = st.selectbox(
                "Residence", list(origin_map.keys()),
                help="Urban or Rural origin"
            )
        with c2:
            transpl = st.checkbox(
                "Transplant Before Dialysis",
                help="Check if patient had kidney transplant before starting PD"
            )

    # Socioeconomic
    with st.expander("💼 Socioeconomic Status", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            schol = st.selectbox(
                "Scholarship Level",
                df['scholarship level '].dropna().unique().tolist(),
                help="Educational attainment"
            )
        with c2:
            indig = st.checkbox(
                "Indigent CNAM Coverage",
                help="Check if patient is under indigent CNAM"
            )

    # Medical History (binary)
    with st.expander("🩺 Medical History", expanded=False):
        c1, c2 = st.columns(2)
        for i, col in enumerate(binary_cols):
            with (c1 if i % 2 == 0 else c2):
                val = st.checkbox(
                    col.replace("_", " ").title(),
                    help=f"Check if patient has {col.replace('_', ' ').lower()}"
                )
                locals()[col] = int(val)

    # Dialysis Parameters
    with st.expander("💧 Dialysis Parameters", expanded=False):
        # Numeric inputs
        c1, c2 = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with (c1 if i % 2 == 0 else c2):
                locals()[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(df[col].mean()),
                    help=f"Enter patient’s {col.replace('_', ' ')}"
                )
        # Multi-category inputs
        c1, c2 = st.columns(2)
        for i, col in enumerate(multi_cat_cols):
            with (c1 if i % 2 == 0 else c2):
                locals()[col] = st.selectbox(
                    col.strip(),
                    df[col].dropna().unique().tolist(),
                    help=f"Select patient’s {col.strip().lower()}"
                )

    submitted = st.form_submit_button("🔍 Predict")

# ── 7) Predict & Interpret ────────────────────────────────────────────────────
if submitted:
    # Gather inputs into dict
    inp = {
        'Age': age,
        gender_col: gender_map[gender],
        origin_col: origin_map[origin],
        'transplant_before_dialysis': int(transpl),
        'scholarship level ': schol,
        'Indigent_Coverage_CNAM': int(indig)
    }
    # Medical history
    for col in binary_cols:
        inp[col] = locals()[col]
    # Dialysis numeric
    for col in numeric_cols:
        inp[col] = locals()[col]
    # Dialysis multi-cat
    for col in multi_cat_cols:
        inp[col] = locals()[col]

    # Build DataFrame, encode & scale
    input_df = pd.DataFrame([inp])
    for col in multi_cat_cols:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))
    input_scaled = scaler.transform(input_df[X.columns])

    # Predict
    pred = clf.predict(input_scaled)[0]
    st.success(f"**Predicted Technique Survival Level: {pred}**")

    # Add brief interpretation
    if pred == 2:
        st.info("✅ This technique is predicted to succeed for **at least 2 years** (good outcome).")
    else:
        st.warning("⚠️ This technique is predicted **not** to succeed beyond 2 years.")
