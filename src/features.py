# src/features.py
import logging
import pandas as pd
from src.config import DATA_INTERIM, DATA_PROCESSED
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IN_PATH = DATA_INTERIM / "clean.csv"
OUT_PATH = DATA_PROCESSED / "features.csv"

def load_interim():
    logging.info(f"Loading interim data from {IN_PATH}")
    return pd.read_csv(IN_PATH)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # target as binary
    df['y_bin'] = df['y'].map({'no':0, 'yes':1})

    # pdays not contacted
    if 'pdays' in df.columns:
        df['pdays_not_contacted'] = (df['pdays'] == 999).astype(int)

    # log1p balance for modeling
    if 'balance' in df.columns:
        # ensure positivity for log: shift min to 0
        min_bal = df['balance'].min()
        df['balance_log1p'] = np.log1p(df['balance'] - min_bal + 1)

    # campaign buckets
    if 'campaign' in df.columns:
        df['campaign_bucket'] = pd.cut(
            df['campaign'],
            bins=[-1,0,1,2,4,10,100],
            labels=['0','1','2','3-4','5-10','10+']
        ).astype(str)

    return df

def main():
    df = load_interim()
    df = create_features(df)
    logging.info(f"Saving features to {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()
