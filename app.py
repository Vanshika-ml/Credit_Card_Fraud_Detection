import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------
# LOAD FILES
# -----------------------
model = joblib.load("fraud_model.pkl")
cols = joblib.load("columns.pkl")
encoders = joblib.load("encoders.pkl")

st.title("💳 Credit Card Fraud Detection System")
st.caption("Machine Learning based Fraud Detection using Random Forest")

st.sidebar.title("💳 Credit Card Fraud Detection")

st.sidebar.info("""
Machine Learning Model:
✔ Random Forest

Dataset:
✔ Kaggle Credit Card Fraud Dataset

Developer:
Vanshika Varshney
""")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.write("### Original Data")
    st.write(df.head())

    # -----------------------
    # SAFE DROP
    # -----------------------
    drop_cols = [
        "trans_date_trans_time","first","last","street",
        "city","state","job","dob","trans_num","is_fraud"
    ]

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # -----------------------
    # ENCODING (IMPORTANT FIX)
    # -----------------------
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    # -----------------------
    # ALIGN COLUMNS
    # -----------------------
    df = df[cols]

    # -----------------------
    # PREDICTION
    # -----------------------
    preds = model.predict(df)
    prob = model.predict_proba(df)
    df["Confidence(%)"]=(prob.max(axis=1)*100).round(2)

    df["Prediction"] = preds
    df["Result"] = df["Prediction"].apply(
        lambda x: "🚨 FRAUD" if x == 1 else "✅ LEGIT"
    )
    

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="📥Download Prediction Results",data=csv,file_name ="fraud_predictions.csv", mime="text/csv")
    option = st.selectbox(
      "Filter Results",
      ["All", "Fraud Only", "Legit Only"]
    )

    if option == "Fraud Only":
      df = df[df["Prediction"] == 1]

    elif option == "Legit Only":
       df = df[df["Prediction"] == 0]
   
    st.write("### Results")
    st.write(df)
    st.success("Prediction Completed Successfully✅")

    # -----------------------
    # SUMMARY
    # -----------------------
    total = len(df)
    fraud = sum(preds)
    legit = total - fraud
    fraud_percent =(fraud/total)*100
    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Cases", fraud)
    col3.metric("Legit Cases", legit)
    col4.metric("Fraud Percentage",f"{fraud_percent:.2f}%")

    # -----------------------
    # GRAPH
    # -----------------------
    fig, ax = plt.subplots()
    ax.bar(["Legit", "Fraud"], [legit, fraud])
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(5,5))

    ax2.pie(
       [legit, fraud],
       labels=["Legit", "Fraud"],
       autopct="%1.2f%%",
       startangle=90
    )

    ax2.set_title("Fraud vs Legit Transactions")

    st.pyplot(fig2)