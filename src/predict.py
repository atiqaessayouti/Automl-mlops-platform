
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"


def load_pipeline() -> dict:
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            "Modèle absent. Exécutez d'abord : python src/train.py"
        )

    return joblib.load(PIPELINE_PATH)


def predict_churn(client_data: dict) -> dict:
    artifact = load_pipeline()
    df = pd.DataFrame([client_data])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    X_processed = artifact["preprocessor"].transform(df)
    prob = float(
        artifact["model"].predict_proba(X_processed)[0, 1]
    )
    prediction = int(prob >= artifact["threshold"])

    return {
        "model": artifact["model_name"],
        "churn_probability": round(prob, 4),
        "threshold": artifact["threshold"],
        "churn_prediction": prediction,
        "risk_label": (
            "High Risk (Churn)"
            if prediction == 1
            else "Low Risk (Stay)"
        ),
    }
