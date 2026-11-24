import math
import time

import pandas as pd
import streamlit as st

from app.data_access import (
    API_URL,
    MLFLOW_DB_PATH,
    PROMETHEUS_URL,
    check_prometheus,
    load_api_health,
    load_mlflow_snapshot,
    query_prometheus,
    query_prometheus_range,
)
from app.ui import metric_card, page_header, status_card


def metric_time_series(
    samples: list[list[object]], value_label: str, multiplier: float = 1.0
) -> pd.DataFrame:
    rows = []
    for sample in samples:
        if len(sample) < 2:
            continue
        try:
            timestamp_value = float(sample[0])
            value = float(sample[1]) * multiplier
        except (OverflowError, TypeError, ValueError):
            continue
        if not math.isfinite(timestamp_value) or not math.isfinite(value):
            continue
        try:
            timestamp = pd.to_datetime(timestamp_value, unit="s", utc=True)
        except (OverflowError, TypeError, ValueError):
            continue
        rows.append((timestamp, value))
    return pd.DataFrame(rows, columns=["Heure (UTC)", value_label])


page_header(
    "Supervision",
    "État des services et télémétrie des requêtes depuis FastAPI et Prometheus. Les indicateurs viennent des services actifs.",
    ":material/monitoring:",
)

api_health, api_error = load_api_health()
prometheus_healthy, prometheus_error = check_prometheus()
mlflow, mlflow_error = load_mlflow_snapshot()

total_results, total_error = query_prometheus('sum(http_requests_total{job="automl-api"})')
rate_results, rate_error = query_prometheus(
    'sum(rate(http_requests_total{job="automl-api"}[5m]))'
)
p95_results, p95_error = query_prometheus(
    'histogram_quantile(0.95, sum by (le)(rate(http_request_duration_seconds_bucket{job="automl-api"}[5m])))'
)
target_results, target_error = query_prometheus('up{job="automl-api"}')


def first_value(results: list[dict] | None) -> float | None:
    if not results:
        return None
    value = results[0].get("value")
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return float(value[1])
    except (TypeError, ValueError):
        return None


def metric_value(value: float | None, error: str | None, suffix: str = "") -> str:
    if error:
        return "Indisponible"
    if value is None:
        return "Aucune donnée"
    return f"{value:,.2f}{suffix}"


service_columns = st.columns(3)
with service_columns[0]:
    status_card(
        "FastAPI",
        bool(api_health and api_health.get("status") == "healthy"),
        f"{API_URL} · modèle {api_health.get('model', 'non renseigné')}"
        if api_health
        else api_error or "État indisponible",
        "monitor-api",
    )
with service_columns[1]:
    status_card(
        "Prometheus",
        prometheus_healthy,
        PROMETHEUS_URL if prometheus_healthy else prometheus_error or "État indisponible",
        "monitor-prometheus",
    )
with service_columns[2]:
    status_card(
        "Métadonnées MLflow",
        mlflow is not None,
        f"{mlflow['total_runs']} essais dans SQLite"
        if mlflow
        else mlflow_error or str(MLFLOW_DB_PATH),
        "monitor-mlflow",
    )

with st.grid(4, min_column_width=180, gap="medium"):
    metric_card(
        "Requêtes API au total",
        metric_value(first_value(total_results), total_error),
        "Somme du compteur Prometheus" if not total_error else total_error,
        "monitor-total-requests",
        "bar_chart",
    )
    metric_card(
        "Débit des requêtes",
        metric_value(first_value(rate_results), rate_error, " req/s"),
        "Débit moyen sur 5 minutes" if not rate_error else rate_error,
        "monitor-request-rate",
        "speed",
    )
    metric_card(
        "Latence p95",
        metric_value(first_value(p95_results) * 1000 if first_value(p95_results) is not None else None, p95_error, " ms"),
        "95e percentile de l’histogramme" if not p95_error else p95_error,
        "monitor-latency",
        "timer",
    )
    target_value = first_value(target_results)
    metric_card(
        "Collecte de l’API",
        "ACTIF" if target_value == 1 else "INACTIF" if target_value == 0 else "—",
        "État de la cible Prometheus" if not target_error else target_error,
        "monitor-scrape-status",
        "cloud_sync",
    )

st.subheader("Télémétrie des requêtes · dernière heure", anchor=False)
end_time = int(time.time())
start_time = end_time - 3600
rate_series, rate_range_error = query_prometheus_range(
    'sum(rate(http_requests_total{job="automl-api"}[5m]))',
    start_time,
    end_time,
)
latency_series, latency_range_error = query_prometheus_range(
    'histogram_quantile(0.95, sum by (le)(rate(http_request_duration_seconds_bucket{job="automl-api"}[5m])))',
    start_time,
    end_time,
)
chart_columns = st.columns(2, gap="large")
with chart_columns[0]:
    with st.container(border=True, key="panel-request-rate-chart"):
        st.markdown("#### Requêtes par seconde")
        if rate_series:
            samples = rate_series[0].get("values", [])
            rate_data = metric_time_series(samples, "Requêtes par seconde")
            if not rate_data.empty:
                st.line_chart(
                    rate_data,
                    x="Heure (UTC)",
                    y="Requêtes par seconde",
                    alt="Débit des requêtes API collecté par Prometheus au cours de la dernière heure",
                )
            else:
                st.info("Prometheus n’a pas collecté de débit de requêtes sur cette période.")
        elif rate_range_error:
            st.warning(rate_range_error)
        else:
            st.info("Aucune donnée de débit n’est disponible sur cette période.")
with chart_columns[1]:
    with st.container(border=True, key="panel-latency-chart"):
        st.markdown("#### Latence p95 des requêtes")
        if latency_series:
            samples = latency_series[0].get("values", [])
            latency_data = metric_time_series(samples, "Latence (ms)", multiplier=1000)
            if not latency_data.empty:
                st.line_chart(
                    latency_data,
                    x="Heure (UTC)",
                    y="Latence (ms)",
                    alt="Latence API au 95e percentile collectée par Prometheus durant la dernière heure",
                )
            else:
                st.info("Prometheus n’a pas collecté de mesures de latence sur cette période.")
        elif latency_range_error:
            st.warning(latency_range_error)
        else:
            st.info("Aucune donnée de latence n’est disponible sur cette période.")

st.caption(
    "MLflow utilise une base de métadonnées SQLite locale ; aucun serveur de suivi MLflow n’est configuré dans Compose."
)
