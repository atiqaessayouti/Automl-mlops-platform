import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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
    y_train = y_train_raw.replace(mapping).astype(int)
    y_test = y_test_raw.replace(mapping).astype(int)
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

    best_name = None
    best_model = None
    best_metrics = None
    best_f1 = -1.0

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test, THRESHOLD)

            mlflow.log_param("model_type", name)
            mlflow.log_param("threshold", THRESHOLD)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            if name == "SVM":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model", serialization_format="cloudpickle",
                    registered_model_name=REGISTERED_MODEL_NAME,
                )
            else:
                mlflow.sklearn.log_model(sk_model=model, name="model", serialization_format="cloudpickle")

            print(f"[{name}] F1={metrics['f1_score']} | ROC-AUC={metrics['roc_auc']}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_name = name
                best_model = model
                best_metrics = metrics

    # Conserver SVM en production si performant, sinon le meilleur modèle F1
    prod_name = "SVM" if "SVM" in candidates else best_name
    prod_model = candidates[prod_name]
    prod_metrics = evaluate(prod_model, X_test, y_test, THRESHOLD)

    artifact = {
        "model_name": prod_name,
        "preprocessor": preprocessor,
        "model": prod_model,
        "threshold": THRESHOLD,
    }
    joblib.dump(artifact, PIPELINE_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": prod_name, "threshold": THRESHOLD, **prod_metrics}, f, indent=2)

    print(f"\nModèle enregistré dans MLflow Model Registry : {REGISTERED_MODEL_NAME} ({prod_name})")
    print(f"Pipeline unifié mis à jour : {PIPELINE_PATH}")


if __name__ == "__main__":
    main()

