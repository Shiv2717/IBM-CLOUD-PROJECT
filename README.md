# NetGuard — Network Intrusion Detection System

An interactive machine-learning application for classifying network traffic as normal or potentially malicious. This project extends the original IBM Cloud Decision Tree notebook into a reusable web application.

## Features

- CSV dataset upload
- Automatic numeric/categorical preprocessing
- Decision Tree and Random Forest models
- Train/test evaluation
- Accuracy, precision, recall and F1 score
- Confusion matrix and classification report
- Interactive single-record prediction
- Confidence/probability display
- Streamlit web interface

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, Jupyter Notebook, IBM Cloud.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a labelled network-traffic CSV and select its target/label column. The app uses the uploaded data to train the selected classifier.

## Project structure

- `app.py` — interactive application
- `requirements.txt` — Python dependencies
- `_P2 - Snap Decision Tree Classifier_ networking.ipynb` — original ML work
- `Network Intrusion Detection.pdf` — original project report

## Security note

This is an academic/demo IDS. It should not be treated as a production security control without representative data, rigorous validation, monitoring, and security review.
