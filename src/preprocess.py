# src/preprocess.py
import logging
import joblib
import pandas as pd
from pathlib import Path
from src.config import DATA_PROCESSED, MODELS_DIR, RANDOM_SEED
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IN_PATH = DATA_PROCESSED / "features.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

def build_preprocessor(df: pd.DataFrame):
    # choose features: exclude target and duration
    exclude = {'y','y_bin','duration'}
    features = [c for c in df.columns if c not in exclude]
    # identify categorical and numeric
    categorical = [c for c in features if df[c].dtype == 'object' or c.endswith('_bucket')]
    numeric = [c for c in features if np.issubdtype(df[c].dtype, np.number)]

    logging.info(f"Numeric features: {numeric}")
    logging.info(f"Categorical features: {categorical}")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ], remainder="drop")
    return preprocessor, numeric, categorical

def main():
    df = pd.read_csv(IN_PATH)
    preprocessor, numeric, categorical = build_preprocessor(df)
    # fit preprocessor
    logging.info("Fitting preprocessor")
    preprocessor.fit(df)
    logging.info(f"Saving preprocessor to {PREPROCESSOR_PATH}")
    joblib.dump({
        'preprocessor': preprocessor,
        'numeric': numeric,
        'categorical': categorical
    }, PREPROCESSOR_PATH)

if __name__ == "__main__":
    main()
