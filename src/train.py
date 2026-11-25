import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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
RANDOM_STATE = 42
OPTUNA_TRIALS_PER_MODEL = 10
CV_SPLITS = 3


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


def _suggest_params(model_name, trial):
    if model_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["none", "balanced"]
            ),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", ["none", "5", "10", "20"]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", "none"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["none", "balanced", "balanced_subsample"]
            ),
        }
    if model_name == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if model_name == "SVM":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        params = {
            "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
            "kernel": kernel,
            "class_weight": trial.suggest_categorical(
                "class_weight", ["none", "balanced"]
            ),
        }
        if kernel == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        return params
    raise ValueError(f"Modèle AutoML non pris en charge : {model_name}")


def _create_model(model_name, params=None):
    params = params or {}
    class_weight = params.get("class_weight")
    if class_weight == "none":
        class_weight = None

    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            class_weight=class_weight,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    if model_name == "RandomForest":
        max_depth = params.get("max_depth")
        if max_depth == "none":
            max_depth = None
        max_features = params.get("max_features")
        if max_features == "none":
            max_features = None
        elif max_features is None:
            max_features = "sqrt"
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=int(max_depth) if max_depth is not None else None,
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=max_features,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            subsample=params.get("subsample", 1.0),
            random_state=RANDOM_STATE,
        )
    if model_name == "SVM":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            class_weight=class_weight,
            probability=True,
            random_state=RANDOM_STATE,
            cache_size=256,
        )
    raise ValueError(f"Modèle AutoML non pris en charge : {model_name}")


def tune_candidate(model_name, X_fit, y_fit, n_trials=OPTUNA_TRIALS_PER_MODEL):
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial):
        params = _suggest_params(model_name, trial)
        fold_scores = []
        for train_indices, validation_indices in cv.split(X_fit, y_fit):
            model = _create_model(model_name, params)
            X_fold_train = X_fit.iloc[train_indices]
            X_fold_validation = X_fit.iloc[validation_indices]
            y_fold_train = y_fit.iloc[train_indices]
            y_fold_validation = y_fit.iloc[validation_indices]
            model.fit(X_fold_train, y_fold_train)
            probabilities = model.predict_proba(X_fold_validation)[:, 1]
            predictions = (probabilities >= THRESHOLD).astype(int)
            fold_scores.append(
                f1_score(y_fold_validation, predictions, zero_division=0)
            )
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=f"{model_name}_optuna",
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return _create_model(model_name, study.best_params), study


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    best_name = None
    best_model = None
    best_f1 = -1.0
    best_study = None
    trial_records = []
    best_params_report = {}

    candidate_names = (
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "SVM",
    )
    for name in candidate_names:
        model, study = tune_candidate(name, X_fit, y_fit)
        model.fit(X_fit, y_fit)
        validation_metrics = evaluate(model, X_validation, y_validation, THRESHOLD)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_param("threshold", THRESHOLD)
            mlflow.log_param("selection_metric", "validation_f1_score")
            mlflow.log_param("optimization_method", "Optuna_TPE")
            mlflow.log_param("optuna_trials", OPTUNA_TRIALS_PER_MODEL)
            mlflow.log_param("cv_strategy", "StratifiedKFold")
            mlflow.log_param("cv_splits", CV_SPLITS)
            mlflow.log_param("optuna_best_cv_f1_score", study.best_value)
            mlflow.log_params(
                {
                    f"optuna_best_{key}": str(value)
                    for key, value in study.best_params.items()
                }
            )
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(
                {
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                }
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
            )

            print(
                f"[{name}] CV F1={study.best_value:.4f} | "
                f"Validation F1={validation_metrics['f1_score']:.4f} | "
                f"ROC-AUC={validation_metrics['roc_auc']:.4f}"
            )

            if validation_metrics["f1_score"] > best_f1:
                best_f1 = validation_metrics["f1_score"]
                best_name = name
                best_model = model
                best_study = study

        best_params_report[name] = {
            "best_trial": study.best_trial.number,
            "best_cv_f1_score": float(study.best_value),
            "best_params": study.best_params,
            "validation_f1_score": validation_metrics["f1_score"],
        }
        for trial in study.trials:
            trial_records.append(
                {
                    "model": name,
                    "trial": trial.number,
                    "state": trial.state.name,
                    "cv_f1_score": (
                        float(trial.value) if trial.value is not None else None
                    ),
                    "duration_seconds": (
                        trial.duration.total_seconds()
                        if trial.duration is not None
                        else None
                    ),
                    "params_json": json.dumps(trial.params, sort_keys=True),
                }
            )

    if best_model is None or best_study is None:
        raise RuntimeError("Aucun modèle n'a pu être entraîné et évalué.")

    best_model.fit(X_train, y_train)
    best_metrics = evaluate(best_model, X_test, y_test, THRESHOLD)

    with mlflow.start_run(run_name=f"{best_name}-registered"):
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("selection_metric", "validation_f1_score")
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_metric("validation_f1_score", best_f1)
        mlflow.log_metric("optuna_best_cv_f1_score", best_study.best_value)
        mlflow.log_param("optimization_method", "Optuna_TPE")
        mlflow.log_param("optuna_trials", OPTUNA_TRIALS_PER_MODEL)
        mlflow.log_params(
            {
                f"optuna_best_{key}": str(value)
                for key, value in best_study.best_params.items()
            }
        )
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

    metrics_report = {
        "model": best_name,
        "threshold": THRESHOLD,
        **best_metrics,
        "selection_metric": "validation_f1_score",
        "validation_f1_score": round(float(best_f1), 4),
        "optimization_method": "Optuna_TPE",
        "optuna_trials_per_model": OPTUNA_TRIALS_PER_MODEL,
        "optuna_cv_splits": CV_SPLITS,
        "optuna_best_cv_f1_score": round(float(best_study.best_value), 4),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    pd.DataFrame(trial_records).to_csv(
        REPORTS_DIR / "optuna_trials.csv",
        index=False,
    )
    with open(REPORTS_DIR / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_report, f, indent=2)

    print(f"\nMeilleur modèle enregistré dans MLflow Model Registry : {REGISTERED_MODEL_NAME} ({best_name})")
    print(f"Pipeline unifié mis à jour : {PIPELINE_PATH}")


if __name__ == "__main__":
    main()

