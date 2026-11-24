import pandas as pd
import streamlit as st

from app.data_access import load_metrics_report, load_mlflow_snapshot
from app.ui import format_decimal, format_timestamp, page_header


page_header(
    "Expériences AutoML",
    "Comparez les modèles candidats selon leurs résultats MLflow. L’évaluation finale est présentée séparément de la validation.",
    ":material/science:",
)

snapshot, snapshot_error = load_mlflow_snapshot()
report, report_error = load_metrics_report()

if snapshot is None:
    st.warning(snapshot_error or "Les métadonnées des expériences MLflow sont indisponibles.")
    st.stop()

if not snapshot["candidate_runs"]:
    st.info("Aucun essai de comparaison terminé n’a été trouvé dans l’expérience Telco-Churn-MLOps.")
else:
    comparisons = pd.DataFrame(snapshot["candidate_runs"])
    metric_columns = [column for column in ("accuracy", "precision", "recall", "f1_score", "roc_auc") if column in comparisons]
    if not metric_columns:
        st.warning("Les essais terminés ne contiennent pas de métriques d’évaluation.")
    else:
        if "f1_score" in comparisons:
            comparisons = comparisons.sort_values("f1_score", ascending=False)
            best_validation = comparisons.iloc[0]
            st.success(
                f"Meilleur score F1 en validation : **{best_validation['model']}** "
                f"({format_decimal(best_validation['f1_score'], 4)})"
            )
        display_comparisons = comparisons.copy()
        display_comparisons["Statut de l’essai"] = display_comparisons["status"].replace(
            {"FINISHED": "Terminé", "FAILED": "Échec", "RUNNING": "En cours", "SCHEDULED": "Planifié"}
        )
        display_comparisons["Début (UTC)"] = display_comparisons["start_time"].map(format_timestamp)
        st.dataframe(
            display_comparisons[["model", *metric_columns, "Statut de l’essai", "Début (UTC)"]].rename(
                columns={
                    "model": "Modèle",
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
                "Score F1": st.column_config.ProgressColumn(
                    "Score F1", min_value=0.0, max_value=1.0, format="%.4f"
                ),
                "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
            },
            hide_index=True,
            alt="Métriques de validation du dernier essai MLflow terminé pour chaque modèle candidat",
        )
        chart_metrics = [column for column in ("f1_score", "roc_auc") if column in comparisons]
        if chart_metrics:
            chart_data = comparisons.set_index("model")[chart_metrics].rename(
                columns={"f1_score": "Score F1", "roc_auc": "Aire sous la courbe ROC"}
            )
            st.bar_chart(
                chart_data,
                y_label="Score de validation",
                stack=False,
                alt="Comparaison des scores F1 et de l’aire sous la courbe ROC des modèles candidats",
            )

st.subheader("Historique des essais", anchor=False)
if snapshot["run_history"]:
    history = pd.DataFrame(snapshot["run_history"])
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
        column_config={"Essai": st.column_config.TextColumn("Essai")},
        hide_index=True,
        alt="Historique des derniers essais MLflow et de leurs scores F1",
    )
else:
    st.info("Cette expérience MLflow ne contient pas encore d’essais.")

st.subheader("Évaluation finale sur le jeu de test", anchor=False)
if report:
    st.caption(
        f"Modèle retenu : {report.get('model', 'Non renseigné')} · "
        f"Seuil de décision : {format_decimal(report['threshold'], 2) if 'threshold' in report else 'Non renseigné'}"
    )
    metric_labels = {
        "accuracy": "Exactitude",
        "precision": "Précision",
        "recall": "Rappel",
        "f1_score": "Score F1",
        "roc_auc": "Aire sous la courbe ROC",
    }
    holdout = pd.DataFrame(
        [
            {"Métrique": metric_labels[metric], "Score": report[metric]}
            for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc")
            if metric in report
        ]
    )
    if not holdout.empty:
        st.dataframe(
            holdout,
            column_config={"Score": st.column_config.NumberColumn(format="%.4f")},
            hide_index=True,
            alt="Métriques finales du jeu de test pour le modèle retenu",
        )
else:
    st.warning(report_error or "Le rapport d’évaluation finale est indisponible.")
