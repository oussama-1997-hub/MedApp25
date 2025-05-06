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
# Multicategory features
multi_cat_cols = [
    'scholarship level ', 'Initial_nephropathy',
    'Technique', 'Permeability_type', 'Germ'
]
# Two‐value coded as 1/2
gender_col = 'Gender '
origin_col = 'Rural_or_Urban_Origin'
gender_map = {"Male": 1, "Female": 2}
origin_map = {"Urban": 2, "Rural": 1}

# Numeric: everything else except target and the two we remove from input
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

# Label‐encode multicategory
le_dict = {}
for col in multi_cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

# Prepare X and y
X = df_enc.drop(columns=[target])
y = df_enc[target]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, y)

# ── 5) Header ─────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;color:#2E86C1;'>🎯 Technique Survival Predictor</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#566573;'>"
    "Complete your patient’s details below, then click “Predict” to see if the peritoneal "
    "dialysis technique is likely to succeed ≥2 years."
    "</p>", unsafe_allow_html=True
)

# ── 6) Input Form ──────────────────────────────────────────────────────────────
with st.form("patient_form"):
    # Demographics
    with st.expander("👤 Demographics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=0, max_value=120,
                                  value=int(df['Age'].mean()))
        with c2:
            gender = st.selectbox("Gender", list(gender_map.keys()))
        c1, c2 = st.columns(2)
        with c1:
            origin = st.selectbox("Residence", list(origin_map.keys()))
        with c2:
            transpl = st.checkbox("Transplant Before Dialysis")

    # Socioeconomic
    with st.expander("💼 Socioeconomic Status", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            schol = st.selectbox("Scholarship Level",
                                 df['scholarship level '].dropna().unique().tolist())
        with c2:
            indig = st.checkbox("Indigent CNAM Coverage")

    # Medical History
    with st.expander("🩺 Medical History", expanded=False):
        c1, c2 = st.columns(2)
        for i, col in enumerate(binary_cols):
            with (c1 if i % 2 == 0 else c2):
                val = st.checkbox(col.replace("_", " ").title())
                locals()[col] = int(val)

    # Dialysis Parameters
    with st.expander("💧 Dialysis Parameters", expanded=False):
        # Numeric
        c1, c2 = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with (c1 if i % 2 == 0 else c2):
                locals()[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(df[col].mean())
                )
        # Multicategory
        c1, c2 = st.columns(2)
        for i, col in enumerate(multi_cat_cols):
            with (c1 if i % 2 == 0 else c2):
                locals()[col] = st.selectbox(
                    col.strip(),
                    df[col].dropna().unique().tolist()
                )

    submitted = st.form_submit_button("🔍 Predict")

# ── 7) Predict & Interpret ─────────────────────────────────────────────────────
if submitted:
    # Build input dict
    inp = {}
    for col in X.columns:
        if col == 'Age':
            inp[col] = age
        elif col == gender_col:
            inp[col] = gender_map[gender]
        elif col == origin_col:
            inp[col] = origin_map[origin]
        elif col in numeric_cols:
            inp[col] = locals()[col]
        elif col in binary_cols:
            inp[col] = locals()[col]
        elif col in multi_cat_cols:
            inp[col] = locals()[col]
        else:
            # Just in case
            inp[col] = df[col].mode()[0]

    # DataFrame & encode multi-cat
    input_df = pd.DataFrame([inp])
    for col in multi_cat_cols:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))

    # Reindex and scale
    input_df = input_df.reindex(columns=X.columns)
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = clf.predict(input_scaled)[0]
    st.success(f"**Predicted Technique Survival Level: {pred}**")
    if pred == 2:
        st.info("✅ Likely to succeed for **≥2 years**.")
    else:
        st.warning("⚠️ Unlikely to succeed beyond 2 years.")
