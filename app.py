import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="NetGuard IDS", page_icon="🛡️", layout="wide")

st.title("🛡️ NetGuard — Network Intrusion Detection")
st.caption("Machine-learning powered network traffic classification")

with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("Model", ["Decision Tree", "Random Forest"])
    test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    st.divider()
    st.info("Upload a labelled CSV dataset. The final column is selected as the target by default.")

uploaded = st.file_uploader("Upload network traffic CSV", type=["csv"])

if uploaded is None:
    st.warning("Upload your network dataset to train and evaluate the detector.")
    st.markdown("### Expected workflow")
    st.markdown("1. Upload a labelled CSV\n2. Select the target column\n3. Train the classifier\n4. Review metrics and confusion matrix\n5. Use the trained pipeline for predictions")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("The CSV needs at least one feature column and one target column.")
    st.stop()

st.subheader("Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", len(df.columns) - 1)
c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

target = st.selectbox("Target / label column", list(df.columns), index=len(df.columns) - 1)

X = df.drop(columns=[target]).copy()
y_raw = df[target].copy()

# Remove rows with missing target labels.
mask = y_raw.notna()
X = X.loc[mask].reset_index(drop=True)
y_raw = y_raw.loc[mask].reset_index(drop=True)

if y_raw.nunique() < 2:
    st.error("The selected target must contain at least two classes.")
    st.stop()

# Encode target labels while retaining their original names.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.astype(str))

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

transformers = []
if numeric_cols:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric_cols))
if categorical_cols:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)

if model_name == "Decision Tree":
    classifier = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=int(random_state),
    )
else:
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=int(random_state),
    )

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier),
])

stratify = y if np.min(np.bincount(y)) >= 2 else None
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=stratify
    )
except ValueError as exc:
    st.error(f"Could not split the data. Try a larger dataset or different test size. Details: {exc}")
    st.stop()

if st.button("🚀 Train intrusion detector", type="primary", use_container_width=True):
    with st.spinner("Training model..."):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    st.session_state["pipeline"] = pipeline
    st.session_state["metrics"] = (accuracy, precision, recall, f1)
    st.session_state["test_data"] = (X_test, y_test, predictions)
    st.session_state["labels"] = label_encoder.classes_.tolist()
    st.session_state["target"] = target
    st.session_state["probabilities"] = probabilities

if "pipeline" not in st.session_state:
    st.info("Click **Train intrusion detector** to generate results.")
    st.stop()

accuracy, precision, recall, f1 = st.session_state["metrics"]
X_test, y_test, predictions = st.session_state["test_data"]
labels = st.session_state["labels"]

st.subheader("Model performance")
metrics = st.columns(4)
metrics[0].metric("Accuracy", f"{accuracy:.2%}")
metrics[1].metric("Precision", f"{precision:.2%}")
metrics[2].metric("Recall", f"{recall:.2%}")
metrics[3].metric("F1 score", f"{f1:.2%}")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y_test, predictions, labels=range(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.dataframe(cm_df, use_container_width=True)

with right:
    st.markdown("#### Classification report")
    report = classification_report(
        y_test, predictions, labels=range(len(labels)), target_names=labels,
        output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(3), use_container_width=True)

st.subheader("Live prediction")
st.caption("Provide one network record using the same feature schema as the uploaded dataset.")

with st.form("prediction_form"):
    input_values = {}
    cols = st.columns(2)
    for i, col in enumerate(X.columns):
        default = X[col].dropna().iloc[0] if not X[col].dropna().empty else ""
        with cols[i % 2]:
            if col in categorical_cols:
                options = X[col].dropna().astype(str).unique().tolist()
                options = options[:100]
                input_values[col] = st.selectbox(col, options, index=0) if options else st.text_input(col, value=str(default))
            else:
                try:
                    input_values[col] = st.number_input(col, value=float(default))
                except (TypeError, ValueError):
                    input_values[col] = st.text_input(col, value=str(default))
    submitted = st.form_submit_button("🔎 Analyze traffic", use_container_width=True)

if submitted:
    row = pd.DataFrame([input_values])
    pred_encoded = st.session_state["pipeline"].predict(row)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    probs = st.session_state["pipeline"].predict_proba(row)[0]
    confidence = float(np.max(probs))

    if str(pred_label).lower() in {"normal", "benign", "0"}:
        st.success(f"Prediction: **{pred_label}** — confidence {confidence:.1%}")
    else:
        st.error(f"Prediction: **{pred_label}** — confidence {confidence:.1%}")

st.divider()
st.caption("NetGuard is a demonstration/academic IDS. Validate models against representative security datasets before production use.")
