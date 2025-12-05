# src/evaluate.py
import logging
import joblib
import pandas as pd
from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEATURES_PATH = DATA_PROCESSED / "features.csv"

def load_test():
    df = pd.read_csv(FEATURES_PATH)
    # define test split the same way as train (or load holdout)
    from sklearn.model_selection import train_test_split
    X = df[[c for c in df.columns if c not in ['y','y_bin','duration']]]
    y = df['y_bin']
    _, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    return X_test, y_test

def evaluate_model(pipeline_path, X_test, y_test):
    pipe = joblib.load(pipeline_path)
    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = pipe.predict(X_test)
    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    pr = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(pr[1], pr[0])
    return {'roc_auc': roc, 'pr_auc': pr_auc, 'report': report, 'y_prob': y_prob, 'y_pred': y_pred}

def plot_pr_curve(y_test, y_prob, out_file):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def main():
    X_test, y_test = load_test()
    results = {}
    for p in (MODELS_DIR).glob("*_pipeline.joblib"):
        logging.info(f"Evaluating {p.name}")
        r = evaluate_model(p, X_test, y_test)
        results[p.name] = r
        # save report
        report_df = pd.DataFrame(r['report']).transpose()
        report_df.to_csv(REPORTS_DIR / f"{p.stem}_classification_report.csv")
        # PR plot
        plot_pr_curve(y_test, r['y_prob'], REPORTS_DIR / f"{p.stem}_pr_curve.png")
    # summary
    summary = {k: {'roc_auc': v['roc_auc'], 'pr_auc': v['pr_auc']} for k, v in results.items()}
    pd.DataFrame(summary).T.to_csv(REPORTS_DIR / "model_summary.csv")
    logging.info("Evaluation complete. Reports saved.")

if __name__ == "__main__":
    main()
