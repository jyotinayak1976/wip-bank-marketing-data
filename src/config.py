from pathlib import Path

PROJECT_ROOT = Path(r"D:\development\ML Dev projects\Bank-Marketing-ML-Dev").resolve()

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

for p in (DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42