import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(
    os.getenv(
        "DATASET_PATH",
        str(PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"),
    )
)
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(PROJECT_ROOT / "reports" / "metrics.json")))
MLFLOW_DB_PATH = Path(os.getenv("MLFLOW_DB_PATH", str(PROJECT_ROOT / "mlflow.db")))
PIPELINE_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090").rstrip("/")

METRIC_KEYS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


@st.cache_data(ttl=300, show_spinner=False)
def load_churn_dataset() -> tuple[pd.DataFrame | None, str | None]:
    if not DATASET_PATH.is_file():
        return None, f"Le jeu de données est introuvable à l’emplacement {DATASET_PATH}."
    try:
        return pd.read_csv(DATASET_PATH), None
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        return None, f"Impossible de lire le jeu de données de résiliation : {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def load_metrics_report() -> tuple[dict[str, Any] | None, str | None]:
    if not METRICS_PATH.is_file():
        return None, f"Le rapport de métriques est introuvable à l’emplacement {METRICS_PATH}."
    try:
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            report = json.load(metrics_file)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Impossible de lire le rapport de métriques : {exc}"
    if not isinstance(report, dict):
        return None, "Le format du rapport de métriques est inattendu."
    return report, None


@st.cache_data(ttl=60, show_spinner=False)
def load_mlflow_snapshot() -> tuple[dict[str, Any] | None, str | None]:
    if not MLFLOW_DB_PATH.is_file():
        return None, f"La base de métadonnées MLflow est introuvable à l’emplacement {MLFLOW_DB_PATH}."

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{MLFLOW_DB_PATH.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=2,
        )
        connection.row_factory = sqlite3.Row
        experiment = connection.execute(
            """
            SELECT experiment_id
            FROM experiments
            WHERE name = ? AND lifecycle_stage = 'active'
            ORDER BY creation_time DESC
            LIMIT 1
            """,
            ("Telco-Churn-MLOps",),
        ).fetchone()
        if experiment is None:
            return {
                "candidate_runs": [],
                "run_history": [],
                "registry_models": [],
                "registry_versions": [],
                "total_runs": 0,
                "latest_run": None,
            }, None

        experiment_id = experiment["experiment_id"]
        run_rows = connection.execute(
            """
            SELECT run_uuid, name, status, start_time, end_time
            FROM runs
            WHERE experiment_id = ? AND lifecycle_stage = 'active'
            ORDER BY start_time DESC
            """,
            (experiment_id,),
        ).fetchall()
        run_ids = [row["run_uuid"] for row in run_rows]
        metrics_by_run: dict[str, dict[str, float]] = {}
        params_by_run: dict[str, dict[str, str]] = {}

        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            for row in connection.execute(
                f"""
                SELECT run_uuid, key, value, timestamp
                FROM metrics
                WHERE run_uuid IN ({placeholders})
                ORDER BY timestamp
                """,
                run_ids,
            ):
                metrics_by_run.setdefault(row["run_uuid"], {})[row["key"]] = float(row["value"])

            for row in connection.execute(
                f"""
                SELECT run_uuid, key, value
                FROM params
                WHERE run_uuid IN ({placeholders})
                """,
                run_ids,
            ):
                params_by_run.setdefault(row["run_uuid"], {})[row["key"]] = row["value"]

        candidate_by_model: dict[str, dict[str, Any]] = {}
        for run in run_rows:
            run_name = run["name"] or "Essai sans nom"
            if run_name.endswith("-registered"):
                continue
            model_name = params_by_run.get(run["run_uuid"], {}).get("model_type", run_name)
            if model_name in candidate_by_model:
                continue
            candidate_by_model[model_name] = {
                "model": model_name,
                "run_id": run["run_uuid"],
                "status": run["status"],
                "start_time": run["start_time"],
                **metrics_by_run.get(run["run_uuid"], {}),
            }

        run_history = [
            {
                "run_name": row["name"] or "Essai sans nom",
                "run_id": row["run_uuid"],
                "status": row["status"],
                "start_time": row["start_time"],
                **metrics_by_run.get(row["run_uuid"], {}),
            }
            for row in run_rows[:12]
        ]
        latest_run = run_history[0] if run_history else None

        registered_models = [
            dict(row)
            for row in connection.execute(
                """
                SELECT name, creation_time, last_updated_time, description
                FROM registered_models
                ORDER BY last_updated_time DESC
                """
            )
        ]
        registry_versions = []
        for row in connection.execute(
            """
            SELECT name, version, current_stage, status, creation_time, run_id
            FROM model_versions
            ORDER BY creation_time DESC
            """
        ):
            version = dict(row)
            version["metrics"] = metrics_by_run.get(version["run_id"], {})
            registry_versions.append(version)

        total_runs = connection.execute(
            """
            SELECT COUNT(*)
            FROM runs
            WHERE experiment_id = ? AND lifecycle_stage = 'active'
            """,
            (experiment_id,),
        ).fetchone()[0]

        return {
            "candidate_runs": list(candidate_by_model.values()),
            "run_history": run_history,
            "registry_models": registered_models,
            "registry_versions": registry_versions,
            "total_runs": total_runs,
            "latest_run": latest_run,
        }, None
    except sqlite3.Error as exc:
        return None, f"Impossible d’interroger la base de métadonnées MLflow : {exc}"
    finally:
        if connection is not None:
            connection.close()


@st.cache_data(ttl=20, show_spinner=False)
def load_api_health() -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = httpx.get(f"{API_URL}/health", timeout=3.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return None, f"La vérification de santé de FastAPI a échoué : {exc}"
    if not isinstance(payload, dict) or "status" not in payload:
        return None, "FastAPI a renvoyé une réponse de santé inattendue."
    return payload, None


@st.cache_data(ttl=15, show_spinner=False)
def check_prometheus() -> tuple[bool, str | None]:
    try:
        response = httpx.get(f"{PROMETHEUS_URL}/-/healthy", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"La vérification de santé de Prometheus a échoué : {exc}"
    return True, None


@st.cache_data(ttl=15, show_spinner=False)
def query_prometheus(query: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        response = httpx.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return None, f"La requête Prometheus a échoué : {exc}"
    if payload.get("status") != "success":
        return None, "Prometheus a indiqué un échec de la requête."
    results = payload.get("data", {}).get("result")
    if not isinstance(results, list):
        return None, "Prometheus a renvoyé un résultat de requête inattendu."
    return results, None


@st.cache_data(ttl=30, show_spinner=False)
def query_prometheus_range(
    query: str, start: int, end: int, step: int = 60
) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        response = httpx.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=8.0,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return None, f"La requête Prometheus sur une plage temporelle a échoué : {exc}"
    if payload.get("status") != "success":
        return None, "Prometheus a indiqué un échec de la requête sur la plage temporelle."
    results = payload.get("data", {}).get("result")
    if not isinstance(results, list):
        return None, "Prometheus a renvoyé un résultat de plage temporelle inattendu."
    return results, None
