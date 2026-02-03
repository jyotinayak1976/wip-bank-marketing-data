import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "logistic_pipeline.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")

MODEL = joblib.load(MODEL_PATH)
PREPROCESSOR_INFO = joblib.load(PREPROCESSOR_PATH)
FEATURE_COLUMNS = PREPROCESSOR_INFO["numeric"] + PREPROCESSOR_INFO["categorical"]

app = FastAPI(title="Bank Marketing Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


class PredictionRequest(BaseModel):
    data: Dict[str, Any]


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pdays" in df.columns:
        df["pdays_not_contacted"] = (df["pdays"] == 999).astype(int)

    if "balance" in df.columns:
        min_bal = df["balance"].min()
        df["balance_log1p"] = np.log1p((df["balance"] - min_bal + 1).clip(lower=0))

    if "campaign" in df.columns:
        df["campaign_bucket"] = pd.cut(
            df["campaign"],
            bins=[-1, 0, 1, 2, 4, 10, 100],
            labels=["0", "1", "2", "3-4", "5-10", "10+"],
        ).astype(str)

    return df


def build_feature_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df = apply_feature_engineering(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[FEATURE_COLUMNS]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> Dict[str, Any]:
    try:
        features = build_feature_frame(request.data)
        probabilities = MODEL.predict_proba(features)[0]
        prediction = int(probabilities[1] >= 0.5)
        return {
            "prediction": prediction,
            "probability_no": float(probabilities[0]),
            "probability_yes": float(probabilities[1]),
        }
    except Exception as exc:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc))
