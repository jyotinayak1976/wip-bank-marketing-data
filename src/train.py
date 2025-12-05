# src/train.py
import logging
import joblib
import pandas as pd
from src.config import DATA_PROCESSED, MODELS_DIR, RANDOM_SEED
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IN_PATH = DATA_PROCESSED / "features.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

def load_data():
    logging.info(f"Reading features from {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    # drop rows with missing target (if any)
    df = df.dropna(subset=['y_bin'])
    return df

def load_preprocessor():
    logging.info(f"Loading preprocessor from {PREPROCESSOR_PATH}")
    obj = joblib.load(PREPROCESSOR_PATH)
    return obj['preprocessor']

def train_models(X_train, y_train):
    models = {}
    # logistic
    models['logistic'] = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    # rf
    models['rf'] = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=RANDOM_SEED)

    # try LGB and XGB if present
    try:
        import lightgbm as lgb
        models['lgb'] = lgb.LGBMClassifier()
    except Exception:
        logging.info("LightGBM not installed")

    try:
        import xgboost as xgb
        models['xgb'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    except Exception:
        logging.info("XGBoost not installed")

    for name, clf in models.items():
        logging.info(f"Training {name}")
        clf.fit(X_train, y_train)
        models[name] = clf
    return models

def main():
    df = load_data()
    preproc = load_preprocessor()

    target = 'y_bin'
    exclude = ['y','y_bin','duration']
    X = df[[c for c in df.columns if c not in exclude]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_SEED)

    # training loop: create pipeline of preprocessor + model
    models = train_models(
        preproc.fit_transform(X_train),  # not used by most models, but we train on transformed data for tree-based; we'll save pipeline instead
        y_train
    )

    # Save pipelines with preprocessor attached
    for name, clf in models.items():
        pipe = Pipeline([('preproc', preproc), ('clf', clf)])
        out = MODELS_DIR / f"{name}_pipeline.joblib"
        logging.info(f"Saving pipeline {name} -> {out}")
        joblib.dump(pipe, out)

        # quick eval
        y_prob = pipe.predict_proba(X_test)[:,1]
        y_pred = pipe.predict(X_test)
        logging.info(f"Evaluation for {name}:\n{classification_report(y_test, y_pred)}")
        logging.info(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == "__main__":
    main()
