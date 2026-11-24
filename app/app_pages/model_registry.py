import pandas as pd
import streamlit as st

from app.data_access import PIPELINE_PATH, load_api_health, load_metrics_report, load_mlflow_snapshot
from app.ui import format_decimal, format_timestamp, metric_card, page_header


page_header(
    "Registre des modèles",
    "Consultez les versions enregistrées dans MLflow et le modèle actuellement signalé par l’API de prédiction.",
    ":material/inventory_2:",
)

snapshot, snapshot_error = load_mlflow_snapshot()
report, report_error = load_metrics_report()
api_health, api_error = load_api_health()

versions = snapshot["registry_versions"] if snapshot else []
models = snapshot["registry_models"] if snapshot else []
with st.grid(3, min_column_width=190, gap="medium"):
    metric_card(
        "Modèles enregistrés",
        str(len(models)) if snapshot else "—",
        "Noms présents dans MLflow" if snapshot else snapshot_error or "Indisponible",
        "registry-model-count",
        "inventory_2",
    )
    metric_card(
        "Versions de modèles",
        str(len(versions)) if snapshot else "—",
        "Versions présentes dans MLflow" if snapshot else snapshot_error or "Indisponible",
        "registry-version-count",
        "layers",
    )
    metric_card(
        "Modèle en service",
        str(api_health.get("model", "—")) if api_health else "—",
        "Nom signalé par l’API en direct" if api_health else api_error or "Indisponible",
        "registry-serving-model",
        "model_training",
    )

if snapshot is None:
    st.warning(snapshot_error or "Les métadonnées du registre MLflow sont indisponibles.")
elif not versions:
    st.info("Aucune version de modèle n’est présente dans le registre MLflow.")
else:
    registry = pd.DataFrame(versions)
    registry["Créée (UTC)"] = registry["creation_time"].map(format_timestamp)
    registry["Étape"] = registry["current_stage"].replace(
        {"None": "Non attribuée", None: "Non attribuée", "Production": "Production", "Staging": "Préproduction"}
    )
    registry["Statut"] = registry["status"].replace(
        {"READY": "Prêt", "PENDING_REGISTRATION": "En attente", "FAILED_REGISTRATION": "Échec"}
    )
    registry["Correspondance API"] = registry["name"].map(
        lambda name: "Modèle signalé par l’API"
        if api_health and api_health.get("model") == name
        else "Non signalé par l’API"
    )
    for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        registry[metric] = registry["metrics"].map(lambda values: values.get(metric))
    st.dataframe(
        registry[
            [
                "name",
                "version",
                "Statut",
                "Étape",
                "Créée (UTC)",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "Correspondance API",
            ]
        ].rename(
            columns={
                "name": "Modèle enregistré",
                "version": "Version",
                "accuracy": "Exactitude",
                "precision": "Précision",
                "recall": "Rappel",
                "f1_score": "Score F1",
                "roc_auc": "ROC-AUC",
            }
        ),
        column_config={
            "Exactitude": st.column_config.NumberColumn(format="%.4f"),
            "Précision": st.column_config.NumberColumn(format="%.4f"),
            "Rappel": st.column_config.NumberColumn(format="%.4f"),
            "Score F1": st.column_config.NumberColumn(format="%.4f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        alt="Versions de modèles MLflow, états, dates de création et métriques enregistrées",
    )
    st.caption(
        "L’endpoint de santé de l’API indique le nom du modèle en service, mais pas sa version ; "
        "aucune version déployée n’est déduite."
    )

st.subheader("Artefact local de déploiement", anchor=False)
artifact_columns = st.columns(2)
with artifact_columns[0]:
    st.badge(
        "Disponible" if PIPELINE_PATH.is_file() else "Manquant",
        color="green" if PIPELINE_PATH.is_file() else "red",
        icon=":material/model_training:",
    )
    st.caption("models/churn_pipeline.joblib")
with artifact_columns[1]:
    if report:
        st.caption(
            f"Modèle évalué : {report.get('model', 'Non renseigné')} · "
            f"score F1 de test : {format_decimal(report['f1_score'], 4) if 'f1_score' in report else 'Non renseigné'}"
        )
    else:
        st.warning(report_error or "Le rapport du modèle final est indisponible.")
