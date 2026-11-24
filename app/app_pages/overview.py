import pandas as pd
import streamlit as st

from app.data_access import (
    PIPELINE_PATH,
    check_prometheus,
    load_api_health,
    load_churn_dataset,
    load_metrics_report,
    load_mlflow_snapshot,
)
from app.ui import (
    format_count,
    format_decimal,
    format_timestamp,
    metric_card,
    page_header,
    status_card,
)


page_header(
    "AutoML Studio",
    "Une plateforme intelligente pour entraîner, évaluer, déployer et superviser des modèles de résiliation client.",
    ":material/dashboard:",
)

report, report_error = load_metrics_report()
mlflow, mlflow_error = load_mlflow_snapshot()
dataset, dataset_error = load_churn_dataset()
api_health, api_error = load_api_health()
prometheus_healthy, prometheus_error = check_prometheus()

candidate_runs = mlflow["candidate_runs"] if mlflow else []
registry_versions = mlflow["registry_versions"] if mlflow else []
best_model = report.get("model") if report else None
test_f1 = report.get("f1_score") if report else None
latest_run = mlflow.get("latest_run") if mlflow else None

with st.grid(4, min_column_width=180, gap="medium"):
    metric_card(
        "Clients dans le jeu de données",
        format_count(len(dataset)) if dataset is not None else "—",
        "Lignes du jeu de données télécom" if dataset is not None else dataset_error or "Indisponible",
        "dataset-records",
        "database",
    )
    metric_card(
        "Modèles comparés",
        str(len(candidate_runs)) if mlflow else "—",
        "Derniers essais MLflow terminés" if mlflow else mlflow_error or "Indisponible",
        "models-compared",
        "science",
    )
    metric_card(
        "Meilleur modèle",
        str(best_model) if best_model else "—",
        "Sélectionné selon le score F1 de validation" if best_model else report_error or "Indisponible",
        "best-model",
        "workspace_premium",
    )
    metric_card(
        "Score F1 final",
        format_decimal(test_f1, 3) if isinstance(test_f1, (int, float)) else "—",
        "Évaluation sur le jeu de test" if report else report_error or "Indisponible",
        "holdout-f1",
        "query_stats",
    )

if latest_run:
    st.caption(f"Dernier essai AutoML : {format_timestamp(latest_run.get('start_time'))}")
elif mlflow_error:
    st.caption(mlflow_error)
else:
    st.caption("Aucun essai AutoML n’est enregistré dans l’expérience MLflow configurée.")

st.subheader("État de la plateforme", anchor=False)
api_model = api_health.get("model", "Modèle non renseigné") if api_health else "Aucune réponse de santé"
status_columns = st.columns(3)
with status_columns[0]:
    status_card(
        "FastAPI",
        bool(api_health and api_health.get("status") == "healthy"),
        f"Modèle en service : {api_model}" if api_health else api_error or "État indisponible",
        "api-health",
    )
with status_columns[1]:
    status_card(
        "Métadonnées MLflow",
        mlflow is not None,
        f"{mlflow['total_runs']} essais enregistrés · {len(registry_versions)} versions au registre"
        if mlflow
        else mlflow_error or "Métadonnées indisponibles",
        "mlflow-health",
    )
with status_columns[2]:
    status_card(
        "Prometheus",
        prometheus_healthy,
        "Point de collecte des métriques accessible"
        if prometheus_healthy
        else prometheus_error or "État indisponible",
        "prometheus-health",
    )

left, right = st.columns([1.2, 1], gap="large")
with left:
    with st.container(border=True, key="panel-model-performance"):
        st.subheader("Performances en validation", anchor=False)
        if candidate_runs:
            comparison = pd.DataFrame(candidate_runs)
            if "f1_score" in comparison.columns:
                comparison = comparison.sort_values("f1_score", ascending=False)
                st.bar_chart(
                    comparison.set_index("model")[["f1_score", "roc_auc"]].rename(
                        columns={"f1_score": "Score F1", "roc_auc": "Aire sous la courbe ROC"}
                    ),
                    y_label="Score de validation",
                    stack=False,
                    alt="Comparaison des scores F1 et de l’aire sous la courbe ROC enregistrés dans MLflow",
                )
            else:
                st.info("Les derniers essais MLflow ne contiennent pas les métriques d’évaluation attendues.")
        elif mlflow_error:
            st.warning(mlflow_error)
        else:
            st.info("Aucun essai de comparaison de modèles n’est disponible pour le moment.")
with right:
    with st.container(border=True, key="panel-featured-model"):
        st.subheader("Modèle retenu", anchor=False)
        if report:
            st.markdown(f"### {report.get('model', 'Nom du modèle non renseigné')}")
            score_columns = st.columns(2)
            with score_columns[0]:
                st.metric(
                    "Aire ROC finale",
                    format_decimal(report["roc_auc"], 3) if "roc_auc" in report else "—",
                )
            with score_columns[1]:
                st.metric(
                    "Seuil de décision",
                    format_decimal(report["threshold"], 2) if "threshold" in report else "—",
                )
            st.caption(
                "Les métriques finales proviennent du jeu de test ; le modèle est sélectionné selon le score F1 de validation."
            )
        else:
            st.warning(report_error or "Aucun rapport d’évaluation finale n’est disponible.")
        st.badge(
            "Pipeline disponible" if PIPELINE_PATH.is_file() else "Pipeline introuvable",
            color="green" if PIPELINE_PATH.is_file() else "red",
            icon=":material/model_training:",
        )

st.subheader("Essais récents", anchor=False)
if mlflow and mlflow["run_history"]:
    history = pd.DataFrame(mlflow["run_history"])
    history["Démarré (UTC)"] = history["start_time"].map(format_timestamp)
    history["Score F1"] = history["f1_score"].map(
        lambda value: f"{value:.4f}" if isinstance(value, (int, float)) else "—"
    )
    history["Statut"] = history["status"].replace(
        {"FINISHED": "Terminé", "FAILED": "Échec", "RUNNING": "En cours", "SCHEDULED": "Planifié"}
    )
    st.dataframe(
        history[["run_name", "Statut", "Démarré (UTC)", "Score F1"]].rename(
            columns={"run_name": "Essai"}
        ),
        column_config={
            "Essai": st.column_config.TextColumn("Essai"),
            "Statut": st.column_config.TextColumn("Statut"),
        },
        hide_index=True,
        alt="Historique récent des essais MLflow terminés",
    )
elif mlflow_error:
    st.warning(mlflow_error)
else:
    st.info("Aucun essai n’est disponible dans l’expérience Telco-Churn-MLOps.")
