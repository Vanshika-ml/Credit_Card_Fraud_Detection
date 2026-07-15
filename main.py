"""
FastAPI service for the Credit Card Fraud Detection model.
Wraps the same preprocessing + prediction logic used in app.py (Streamlit),
but exposes it as a REST API with auto-generated docs (Swagger UI at /docs).
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Random Forest based fraud detection service",
    version="1.0.0"
)

# -----------------------
# LOAD MODEL ARTIFACTS (once, at startup)
# -----------------------
model = joblib.load("fraud_model.pkl")
cols = joblib.load("columns.pkl")
encoders = joblib.load("encoders.pkl")

DROP_COLS = [
    "trans_date_trans_time", "first", "last", "street",
    "city", "state", "job", "dob", "trans_num", "is_fraud"
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Same preprocessing pipeline as the Streamlit app."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[cols]


@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Fraud Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a CSV of transactions and get back fraud predictions.
    Same input format as the Streamlit app's CSV uploader.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    try:
        raw_bytes = await file.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
        original_df = df.copy()

        processed_df = preprocess(df)

        preds = model.predict(processed_df)
        probs = model.predict_proba(processed_df)

        results = []
        for i in range(len(preds)):
            results.append({
                "row": i,
                "prediction": "FRAUD" if preds[i] == 1 else "LEGIT",
                "confidence_percent": round(float(probs[i].max()) * 100, 2)
            })

        fraud_count = int(sum(preds))
        total = len(preds)

        return JSONResponse({
            "summary": {
                "total_transactions": total,
                "fraud_cases": fraud_count,
                "legit_cases": total - fraud_count,
                "fraud_percent": round((fraud_count / total) * 100, 2)
            },
            "results": results
        })

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")