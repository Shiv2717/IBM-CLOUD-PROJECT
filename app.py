import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="NetGuard IDS", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme / helpers ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px; margin-bottom: 1rem;}
    .small-muted {color: #8b949e; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def make_sample_data(n=800, seed=42):
    """Create a safe synthetic labelled network-traffic dataset for demos."""
    rng = np.random.default_rng(seed)
    protocols = rng.choice(["tcp", "udp", "icmp"], n, p=[0.72, 0.20, 0.08])
    services = rng.choice(["http", "https", "ssh", "ftp", "dns", "smtp"], n)
    duration = np.round(rng.exponential(2.5, n), 2)
    src_bytes = rng.lognormal(7.0, 1.25, n).astype(int)
    dst_bytes = rng.lognormal(7.2, 1.30, n).astype(int)
    count = rng.poisson(8, n) + 1
    srv_count = np.maximum(1, count + rng.integers(-3, 4, n))
    failed_logins = rng.poisson(0.35, n)
    error_rate = np.clip(rng.beta(1.2, 8, n), 0, 1)

    risk = (
        (protocols == "icmp").astype(float) * 0.9
        + np.isin(services, ["ftp", "ssh"]).astype(float) * 0.35
        + (count > 13).astype(float) * 1.0
        + (failed_logins >= 2).astype(float) * 1.4
        + (error_rate > 0.25).astype(float) * 1.1
        + (src_bytes > np.percentile(src_bytes, 88)).astype(float) * 0.35
        + rng.normal(0, 0.35, n)
    )
    labels = np.where(risk >= 1.75, "Intrusion", "Normal")
    return pd.DataFrame(
        {
            "protocol": protocols,
            "service": services,
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "count": count,
            "srv_count": srv_count,
            "failed_logins": failed_logins,
            "error_rate": np.round(error_rate, 4),
            "label": labels,
        }
    )


def build_pipeline(X, model_name, random_state):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([( "imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=random_state
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def train_model(df, target, model_name, test_size, random_state):
    X = df.drop(columns=[target]).copy()
    y_raw = df[target].astype(str).copy()
    mask = y_raw.notna()
    X, y_raw = X.loc[mask].reset_index(drop=True), y_raw.loc[mask].reset_index(drop=True)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    if len(encoder.classes_) < 2:
        raise ValueError("The target column must contain at least two classes.")
    counts = np.bincount(y)
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    pipe = build_pipeline(X, model_name, random_state)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }
    return pipe, encoder, X, X_test, y_test, pred, probs, metrics


def feature_importance_df(pipe):
    classifier = pipe.named_steps["classifier"]
    preprocessor = pipe.named_steps["preprocessor"]
    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])
    try:
        names = preprocessor.get_feature_names_out()
        values = classifier.feature_importances_
        out = pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False)
        out["Feature"] = out["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
        return out.head(15).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.selectbox("Model", ["Decision Tree", "Random Forest"])
    test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    st.divider()
    st.markdown("### Quick start")
    if st.button("🧪 Load demo dataset", use_container_width=True):
        st.session_state["df"] = make_sample_data(seed=int(random_state))
        st.session_state.pop("pipeline", None)
        st.session_state["history"] = []
        st.rerun()
    st.caption("The demo dataset is synthetic and safe to use for presentations.")

# ---------- Header ----------
st.markdown(
    '<div class="hero"><h1>🛡️ NetGuard — Network Intrusion Detection</h1>'
    '<p class="small-muted">Machine-learning powered traffic classification, model evaluation and explainable detection.</p></div>',
    unsafe_allow_html=True,
)

# ---------- Dataset input ----------
uploaded = st.file_uploader("Upload a labelled network-traffic CSV", type=["csv"], help="Use a labelled dataset with the prediction target in one column.")
if uploaded is not None:
    try:
        st.session_state["df"] = pd.read_csv(uploaded)
        st.session_state.pop("pipeline", None)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")

if "df" not in st.session_state:
    st.info("👈 Load the demo dataset from the sidebar, or upload your own labelled CSV to begin.")
    st.markdown("### What NetGuard provides")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**🔍 Detection**\n\nClassify network records as normal or suspicious.")
    c2.markdown("**📊 Evaluation**\n\nAccuracy, precision, recall, F1 and confusion matrix.")
    c3.markdown("**🧠 Explainability**\n\nInspect model feature importance and prediction confidence.")
    st.stop()

df = st.session_state["df"].copy()
if df.empty or len(df.columns) < 2:
    st.error("The dataset needs at least one feature column and one target column.")
    st.stop()

# ---------- Dataset overview ----------
st.subheader("📦 Dataset overview")
mc = st.columns(5)
mc[0].metric("Records", f"{len(df):,}")
mc[1].metric("Columns", len(df.columns))
mc[2].metric("Missing", f"{int(df.isna().sum().sum()):,}")
mc[3].metric("Duplicates", f"{int(df.duplicated().sum()):,}")
mc[4].metric("Memory", f"{df.memory_usage(deep=True).sum()/1024**2:.1f} MB")

with st.expander("Preview and data quality", expanded=False):
    st.dataframe(df.head(25), use_container_width=True)
    st.write("**Missing values by column**")
    st.dataframe(df.isna().sum().to_frame("Missing values"), use_container_width=True)

target = st.selectbox("🎯 Target / label column", list(df.columns), index=len(df.columns) - 1)
class_counts = df[target].astype(str).value_counts(dropna=False).rename_axis("Class").reset_index(name="Count")

# ---------- Train ----------
if st.button("🚀 Train intrusion detector", type="primary", use_container_width=True):
    try:
        with st.spinner(f"Training {model_name}..."):
            result = train_model(df, target, model_name, test_size, int(random_state))
        pipe, encoder, X, X_test, y_test, pred, probs, metrics = result
        st.session_state.update(
            {
                "pipeline": pipe,
                "encoder": encoder,
                "X": X,
                "X_test": X_test,
                "y_test": y_test,
                "pred": pred,
                "probs": probs,
                "metrics": metrics,
                "target": target,
                "model_name": model_name,
                "history": [],
            }
        )
        st.success(f"{model_name} trained successfully on {len(X):,} labelled records.")
    except Exception as exc:
        st.error(f"Training failed: {exc}")

if "pipeline" not in st.session_state:
    st.warning("Train the detector to unlock analytics and predictions.")
    st.stop()

# ---------- Navigation ----------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔎 Detection", "🧠 Explainability", "📋 Project Guide"])

metrics = st.session_state["metrics"]
encoder = st.session_state["encoder"]
pipe = st.session_state["pipeline"]
X = st.session_state["X"]
y_test = st.session_state["y_test"]
pred = st.session_state["pred"]
probs = st.session_state["probs"]
labels = encoder.classes_.tolist()

# ---------- Dashboard ----------
with tab1:
    st.subheader("Model performance")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
    cols[1].metric("Precision", f"{metrics['precision']:.2%}")
    cols[2].metric("Recall", f"{metrics['recall']:.2%}")
    cols[3].metric("F1 score", f"{metrics['f1']:.2%}")

    a, b = st.columns(2)
    with a:
        st.markdown("#### Class distribution")
        st.bar_chart(class_counts.set_index("Class"))
    with b:
        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(y_test, pred, labels=range(len(labels)))
        cm_df = pd.DataFrame(cm, index=[f"Actual: {x}" for x in labels], columns=[f"Pred: {x}" for x in labels])
        st.dataframe(cm_df, use_container_width=True)

    st.markdown("#### Classification report")
    report = classification_report(y_test, pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ---------- Detection ----------
with tab2:
    st.subheader("🔎 Traffic detection")
    st.caption("Enter one record using the same feature schema as the training dataset.")
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    with st.form("prediction_form"):
        input_values = {}
        form_cols = st.columns(2)
        for i, col in enumerate(X.columns):
            non_null = X[col].dropna()
            default = non_null.iloc[0] if not non_null.empty else ""
            with form_cols[i % 2]:
                if col in categorical_cols:
                    options = non_null.astype(str).unique().tolist()[:100]
                    input_values[col] = st.selectbox(col, options) if options else st.text_input(col, value=str(default))
                elif pd.api.types.is_numeric_dtype(X[col]):
                    input_values[col] = st.number_input(col, value=float(default) if default != "" else 0.0)
                else:
                    input_values[col] = st.text_input(col, value=str(default))
        submitted = st.form_submit_button("🔍 Analyze traffic", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([input_values])
        pred_encoded = int(pipe.predict(row)[0])
        pred_label = encoder.inverse_transform([pred_encoded])[0]
        row_probs = pipe.predict_proba(row)[0] if hasattr(pipe, "predict_proba") else np.array([1.0])
        confidence = float(np.max(row_probs))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.setdefault("history", []).append(
            {"Timestamp": timestamp, "Prediction": str(pred_label), "Confidence": confidence, **input_values}
        )
        if str(pred_label).lower() in {"normal", "benign", "safe", "0"}:
            st.success(f"🟢 **{pred_label}** — confidence {confidence:.1%}")
        else:
            st.error(f"🔴 **{pred_label}** — confidence {confidence:.1%}")
        st.progress(confidence, text=f"Model confidence: {confidence:.1%}")

    history = pd.DataFrame(st.session_state.get("history", []))
    if not history.empty:
        st.markdown("#### Prediction history")
        st.dataframe(history, use_container_width=True)
        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download prediction history", csv, "netguard_predictions.csv", "text/csv")

    st.markdown("#### Demo traffic")
    st.caption("Generate a synthetic record for a quick presentation/demo. This does not inspect real network packets.")
    if st.button("🎲 Generate demo traffic"):
        sample = make_sample_data(1, seed=int(datetime.now().timestamp()) % 100000)[X.columns]
        demo_pred = encoder.inverse_transform([int(pipe.predict(sample)[0])])[0]
        demo_prob = float(np.max(pipe.predict_proba(sample)[0]))
        st.dataframe(sample, use_container_width=True)
        st.info(f"Demo prediction: **{demo_pred}** ({demo_prob:.1%} confidence)")

# ---------- Explainability ----------
with tab3:
    st.subheader("🧠 Model explainability")
    fi = feature_importance_df(pipe)
    if fi.empty:
        st.info("Feature importance is unavailable for this model.")
    else:
        st.markdown("Top features used by the trained tree-based model")
        st.bar_chart(fi.set_index("Feature")["Importance"])
        st.dataframe(fi, use_container_width=True)
    st.markdown("#### Model configuration")
    st.code(str(pipe.named_steps["classifier"].get_params()), language="text")
    st.info("Feature importance indicates which transformed inputs contributed most to the tree model's decisions. It is not a causal explanation and should be validated before production use.")

# ---------- Project guide ----------
with tab4:
    st.subheader("📋 How NetGuard works")
    st.markdown(
        """
        **1. Data ingestion** → upload a labelled CSV or use the synthetic demo dataset.

        **2. Preprocessing** → numeric missing values are median-imputed; categorical values are most-frequent imputed and one-hot encoded.

        **3. Model training** → choose Decision Tree or Random Forest, then split the data into training and test sets.

        **4. Evaluation** → review accuracy, precision, recall, F1, class distribution and confusion matrix.

        **5. Detection** → enter a network record and receive a class prediction with model confidence.

        **6. Explainability** → inspect tree-based feature importance.
        """
    )
    st.markdown("#### Recommended production architecture")
    st.code(
        "Network traffic → ingestion → preprocessing → ML model → prediction API → database → security dashboard",
        language="text",
    )
    st.warning("NetGuard is an academic/demo IDS. It does not capture packets, block attacks, or replace a production firewall/IDS. Validate models on representative security data before operational use.")

st.divider()
st.caption("NetGuard IDS • Python + Streamlit + scikit-learn • Built as a machine-learning cybersecurity demonstration")
