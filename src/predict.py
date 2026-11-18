import json
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"


def build_and_save_pipeline() -> dict:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "evaluation_models" / "svm.joblib")

    with open(REPORTS_DIR / "evaluation_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    threshold = float(summary["best_by_f1_tuned_threshold"]["Threshold"])
    model_name = summary["best_by_f1_tuned_threshold"]["Model"]

    artifact = {
        "model_name": model_name,
        "preprocessor": preprocessor,
        "model": model,
        "threshold": threshold,
    }
    joblib.dump(artifact, PIPELINE_PATH)
    return artifact


def load_pipeline() -> dict:
    if not PIPELINE_PATH.exists():
        return build_and_save_pipeline()
    return joblib.load(PIPELINE_PATH)


def predict_churn(client_data: dict) -> dict:
    artifact = load_pipeline()
    df = pd.DataFrame([client_data])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X_processed = artifact["preprocessor"].transform(df)
    prob = float(artifact["model"].predict_proba(X_processed)[0, 1])
    prediction = int(prob >= artifact["threshold"])

    return {
        "model": artifact["model_name"],
        "churn_probability": round(prob, 4),
        "threshold": artifact["threshold"],
        "churn_prediction": prediction,
        "risk_label": "High Risk (Churn)" if prediction == 1 else "Low Risk (Stay)",
    }


if __name__ == "__main__":
    build_and_save_pipeline()
    print("Pipeline unifié sauvegardé :", PIPELINE_PATH)

    sample_client = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": 151.65,
    }

    print("Test prédiction :", predict_churn(sample_client))