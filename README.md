# 🛡️ NetGuard — Network Intrusion Detection System

**Live demo:** https://intrusiondetectionin.streamlit.app/

NetGuard is a machine-learning powered Network Intrusion Detection System (NIDS) that classifies labelled network-traffic records as normal or potentially malicious. It extends the original IBM Cloud Decision Tree work into an interactive, presentation-ready web application.

## ✨ What it includes

- 📂 Upload any labelled network-traffic CSV
- 🧪 Built-in synthetic demo dataset — no external dataset required for a presentation
- 🧹 Automatic numeric/categorical preprocessing and missing-value handling
- 🌳 Decision Tree classifier
- 🌲 Random Forest classifier
- 📊 Accuracy, precision, recall and F1 score
- 🔲 Confusion matrix and classification report
- 📈 Class-distribution analytics
- 🔎 Interactive single-record traffic detection
- 🎯 Prediction confidence/probability
- 🧠 Tree-based feature importance / explainability
- 📋 Prediction history
- ⬇️ Download prediction history as CSV
- ☁️ Streamlit cloud deployment
- 📱 Responsive wide-layout dashboard

## 🧠 Machine-learning workflow

```text
Labelled network traffic
          ↓
     Data validation
          ↓
 Missing-value handling
          ↓
 Numeric + categorical preprocessing
          ↓
      Train / test split
          ↓
 Decision Tree / Random Forest
          ↓
     Model evaluation
          ↓
  Traffic classification
          ↓
 Confidence + explainability
```

## 🏗️ Application architecture

```text
              ┌──────────────────┐
              │ CSV / Demo Data  │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │  Preprocessing   │
              └────────┬─────────┘
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
┌─────────────────┐          ┌─────────────────┐
│  Decision Tree  │          │  Random Forest  │
└────────┬────────┘          └────────┬────────┘
         └──────────────┬─────────────┘
                        ↓
              ┌──────────────────┐
              │ Detection Engine │
              └────────┬─────────┘
                       ↓
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
   Dashboard       Prediction       Explainability
                    History         Feature Importance
```

## 🖥️ Dashboard workflow

1. Load the synthetic demo dataset or upload a labelled CSV.
2. Select the target/label column.
3. Choose Decision Tree or Random Forest.
4. Set the test size and random state.
5. Train the intrusion detector.
6. Review model metrics and confusion matrix.
7. Analyze individual network records.
8. Review feature importance and prediction history.
9. Export prediction history as CSV.

## 🧪 Demo dataset

The **Load demo dataset** button creates a synthetic network-traffic dataset inside the application. It is intentionally synthetic so the application can be demonstrated without downloading or exposing real traffic data.

The demo includes features such as protocol, service, duration, source/destination bytes, connection counts, failed logins and error rate, with a generated `Normal` / `Intrusion` label.

**Important:** demo results are for demonstrating the application workflow, not for claiming real-world detection performance.

## 📁 Project structure

```text
IBM-CLOUD-PROJECT/
├── app.py
├── requirements.txt
├── README.md
├── _P2 - Snap Decision Tree Classifier_ networking.ipynb
└── Network Intrusion Detection.pdf
```

## ⚙️ Run locally

```bash
git clone https://github.com/Shiv2717/IBM-CLOUD-PROJECT.git
cd IBM-CLOUD-PROJECT
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deployment

The application is deployed with Streamlit Community Cloud from this GitHub repository.

**Live:** https://intrusiondetectionin.streamlit.app/

Entry point: `app.py`

## 🛠️ Technology stack

- **Language:** Python
- **UI:** Streamlit
- **Data:** pandas, NumPy
- **ML:** scikit-learn
- **Algorithms:** Decision Tree, Random Forest
- **Development:** Jupyter Notebook
- **Original cloud workflow:** IBM Cloud / IBM notebook environment
- **Deployment:** Streamlit Community Cloud


## 🚀 Future production upgrades

- Live packet/flow ingestion using a controlled monitoring environment
- FastAPI prediction service
- MySQL/PostgreSQL prediction store
- Authentication and role-based access
- Alerting through email/webhooks
- Model versioning and retraining pipeline
- SHAP-based explanations
- Docker + CI/CD
- Cloud object storage and monitoring
