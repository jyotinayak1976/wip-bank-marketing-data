# src/data_processing.py
import sys
from pathlib import Path

# Find project root (folder that contains /src)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]   # One level above /src

# Add project root to PYTHONPATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd

from src.config import DATA_RAW, DATA_INTERIM, DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RAW_FILENAME = "bank-full.csv"
OUT_INTERIM = DATA_INTERIM / "clean.csv"
OUT_PROCESSED = DATA_PROCESSED / "initial_clean.csv"

def load_raw():
    path = DATA_RAW / RAW_FILENAME
    logging.info(f"Reading raw data from {path}")
    # UCI bank dataset uses semicolon
    df = pd.read_csv(path, sep=";", quotechar='"')
    return df

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    # trim column names
    df.columns = [c.strip() for c in df.columns]
    # strip whitespace from object columns
    for col in df.select_dtypes(["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Fix known values: some numeric columns may read as strings if quotes were odd
    # convert numeric-like columns
    numeric_cols = ['age','balance','day','duration','campaign','pdays','previous']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Replace some sentinel values e.g., pdays = 999 is common; keep but note later
    return df

def main():
    df = load_raw()
    df = clean_basic(df)
    logging.info("Saving interim cleaned data")
    df.to_csv(OUT_INTERIM, index=False)
    logging.info("Saving processed initial copy")
    df.to_csv(OUT_PROCESSED, index=False)

if __name__ == "__main__":
    main()
