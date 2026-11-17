
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("dataset/creditcard.csv")

model = load_model()
data = load_data()

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div[data-testid="stMetric"] {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
}
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning Based Fraud Detection Dashboard")
st.caption("Powered by Logistic Regression")
st.divider()

# ---------------- DATASET STATISTICS ----------------
total = len(data)
fraud = int(data["Class"].sum())
normal = total - fraud
fraud_percentage = (fraud / total) * 100

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total:,}")
col2.metric("Normal Transactions", f"{normal:,}")
col3.metric("Fraud Transactions", f"{fraud:,}")
col4.metric("Fraud Percentage", f"{fraud_percentage:.3f}%")

st.divider()

# ---------------- TRANSACTION INPUT ----------------
st.subheader("🔍 Check a Transaction")
st.write("Enter transaction details below.")

time = st.number_input(
    "Transaction Time",
    min_value=0.0,
    value=0.0
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

st.subheader("Transaction Features")

features = {}

with st.expander("Enter V1 to V28 Features", expanded=True):
    col1, col2 = st.columns(2)

    for i in range(1, 29):
        with col1 if i % 2 == 1 else col2:
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"feature_{i}"
            )

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Fraud"):

    input_data = {
        "Time": time,
        **features,
        "Amount": amount
    }

    input_df = pd.DataFrame([input_data])

    input_df = input_df[model.feature_names_in_] if hasattr(
        model, "feature_names_in_"
    ) else input_df

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    fraud_probability = probabilities[1] * 100

    if prediction == 1:
        result = "Fraud"
        st.error("⚠️ Potential Fraudulent Transaction Detected!")
    else:
        result = "Non-Fraud"
        st.success("✅ Transaction Classified as Non-Fraudulent")

    st.subheader("Prediction Probability")

    st.progress(float(fraud_probability / 100))

    st.metric(
        "Estimated Fraud Probability",
        f"{fraud_probability:.2f}%"
    )

    st.caption(
        "Model probability is an estimate, not a guarantee of transaction safety."
    )

    st.session_state.history.append({
        "Transaction Time": time,
        "Amount": amount,
        "Prediction": result,
        "Fraud Probability": round(fraud_probability, 2)
    })

# ---------------- PREDICTION HISTORY ----------------
st.divider()
st.subheader("📋 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📥 Download CSV Report",
            data=csv,
            file_name="fraud_prediction_history.csv",
            mime="text/csv"
        )

    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

else:
    st.info("No predictions yet. Make your first prediction!")

# ---------------- TRANSACTION CHART ----------------
st.divider()
st.subheader("📈 Transaction Distribution")

chart_data = pd.DataFrame({
    "Transaction Type": ["Normal", "Fraud"],
    "Count": [normal, fraud]
})

st.bar_chart(
    chart_data.set_index("Transaction Type")
)

# ---------------- FOOTER ----------------
st.divider()

st.markdown("""
### About This Project

This project uses Logistic Regression to classify
credit card transactions into normal and potentially
fraudulent categories.

**Technology Used:**
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

*Educational demonstration only. Not intended for real-world
financial decisions.*
""")

st.caption("Credit Card Fraud Detection | BCA Data Science & AI Project")

st.divider()
st.subheader("🧪 Automatic Fraud Detection Test")

if st.button("Test Sample Fraud Transaction"):
    import pandas as pd

    df = pd.read_csv("dataset/creditcard.csv")
    fraud_sample = df[df["Class"] == 1].iloc[[0]]

    sample_features = fraud_sample.drop(
        columns=[ "Class"]
    )

    prediction = model.predict(sample_features)[0]
    probability = model.predict_proba(sample_features)[0][1]

    st.write("Actual Dataset Label: Fraud")
    st.write(f"Predicted Fraud Probability: {probability:.2%}")

    if prediction == 1:
        st.success("Model Prediction: Fraudulent Transaction")
    else:
        st.warning("Model Prediction: Non-Fraudulent Transaction")
