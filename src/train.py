import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

THRESHOLD = 0.35
EXPERIMENT_NAME = "Telco-Churn-MLOps"
REGISTERED_MODEL_NAME = "TelcoChurnModel"


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train_processed.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_processed.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_test_raw = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    mapping = {"Yes": 1, "No": 0}

    def encode_target(target):
        encoded = pd.to_numeric(target.replace(mapping), errors="raise")
        if encoded.isna().any() or not encoded.isin([0, 1]).all():
            raise ValueError("La cible doit contenir uniquement les valeurs 0 et 1.")
        return encoded.astype(int)

    y_train = encode_target(y_train_raw)
    y_test = encode_target(y_test_raw)
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, threshold=THRESHOLD):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
    }
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    best_name = None
    best_model = None
    best_f1 = -1.0

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_fit, y_fit)
            metrics = evaluate(model, X_validation, y_validation, THRESHOLD)
            selection_f1 = f1_score(
                y_validation,
                (model.predict_proba(X_validation)[:, 1] >= THRESHOLD).astype(int),
                zero_division=0,
            )

            mlflow.log_param("model_type", name)
            mlflow.log_param("threshold", THRESHOLD)
            mlflow.log_param("selection_metric", "f1_score")
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
            )

            print(f"[{name}] F1={metrics['f1_score']} | ROC-AUC={metrics['roc_auc']}")

            if selection_f1 > best_f1:
                best_f1 = selection_f1
                best_name = name
                best_model = model

    best_model.fit(X_train, y_train)
    best_metrics = evaluate(best_model, X_test, y_test, THRESHOLD)

    with mlflow.start_run(run_name=f"{best_name}-registered"):
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("selection_metric", "f1_score")
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_metric("validation_f1_score", best_f1)
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    artifact = {
        "model_name": best_name,
        "preprocessor": preprocessor,
        "model": best_model,
        "threshold": THRESHOLD,
    }
    joblib.dump(artifact, PIPELINE_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": best_name, "threshold": THRESHOLD, **best_metrics}, f, indent=2)

    print(f"\nMeilleur modèle enregistré dans MLflow Model Registry : {REGISTERED_MODEL_NAME} ({best_name})")
    print(f"Pipeline unifié mis à jour : {PIPELINE_PATH}")


if __name__ == "__main__":
    main()

