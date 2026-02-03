# Bank Marketing ML App

This repository contains an end-to-end machine learning workflow for a bank marketing dataset, plus a web-based inference experience. It includes data preparation and model training scripts, a FastAPI inference service that loads the trained pipeline, and a React UI that submits customer profiles and displays predictions.

## What’s inside

### Data & modeling workflow
The ML pipeline is organized in `src/` and follows a typical workflow:

- **Preprocessing**: builds a column transformer with numeric scaling and categorical encoding. It also stores the feature lists required for inference.
- **Feature engineering**: adds engineered fields used by the model (e.g., `pdays_not_contacted`, `balance_log1p`, `campaign_bucket`).
- **Training**: trains multiple models and saves a pipeline that includes the preprocessor + classifier.

The saved artifacts live in `models/`:

- `logistic_pipeline.joblib` — pipeline used by the API (preprocessor + logistic regression).
- `preprocessor.joblib` — stores the fitted preprocessor and the feature column list.
- `rf_grid_best.joblib` — optional model artifact from experiments.

### FastAPI inference service
The API is implemented in `src/api.py` and exposes:

- `GET /health` — returns a basic health check.
- `POST /predict` — accepts a JSON payload with a `data` object of customer features and returns prediction + probabilities.

At startup, the API loads:

- `models/logistic_pipeline.joblib` (the trained pipeline)
- `models/preprocessor.joblib` (feature schema)

Feature engineering is applied at inference time to match training, and any missing fields are filled with nulls before the model is called.

### React web app
The React client lives in `frontend/`. It provides a form with the model’s required inputs and calls the FastAPI `/predict` endpoint. The UI displays:

- The predicted class (Yes/No)
- Probability of subscription (`probability_yes`)
- Probability of no subscription (`probability_no`)

## Repository layout

```
.
├── data/                 # Raw/interim/processed data (CSV)
├── models/               # Serialized ML artifacts (.joblib)
├── notebooks/            # Exploration notebooks
├── reports/              # Figures and outputs
├── src/                  # Training + inference code
│   ├── api.py            # FastAPI inference service
│   ├── preprocess.py     # Preprocessor build + save
│   ├── features.py       # Feature engineering
│   └── train.py          # Model training + pipeline save
├── frontend/             # React app (Vite)
└── requirements.txt      # Backend dependencies
```

## Running the backend (FastAPI)

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the API**
   ```bash
   uvicorn src.api:app --reload --port 8000
   ```

3. **Verify health**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Example prediction request**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "age": 35,
         "job": "admin.",
         "marital": "married",
         "education": "tertiary",
         "default": "no",
         "balance": 1500,
         "housing": "yes",
         "loan": "no",
         "contact": "cellular",
         "day": 5,
         "month": "may",
         "campaign": 2,
         "pdays": 999,
         "previous": 0,
         "poutcome": "unknown"
       }
     }'
   ```

## Running the frontend (React)

1. **Install dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the dev server**
   ```bash
   npm run dev
   ```

3. Open `http://localhost:5173` and submit the form. The app will call `http://localhost:8000/predict`.

## Training the model (optional)

If you want to retrain or regenerate the artifacts:

```bash
python src/preprocess.py
python src/features.py
python src/train.py
```

This will rebuild the preprocessor and overwrite the saved model pipelines under `models/`.

## Notes

- The API expects `models/logistic_pipeline.joblib` and `models/preprocessor.joblib` to be present.
- CORS is enabled for `http://localhost:5173` and `http://localhost:3000` so the React app can access the API locally.
