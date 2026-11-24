import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui import apply_styles

st.set_page_config(
    page_title="AutoML Studio",
    page_icon=":material/model_training:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_styles()

with st.sidebar:
    st.markdown("### :material/model_training: AutoML Studio")
    st.caption("Plateforme intelligente de Machine Learning & MLOps")
    st.badge("Espace résiliation télécom", color="violet", icon=":material/dataset:")

pages = st.navigation(
    {
        "Espace de travail": [
            st.Page(
                str(PROJECT_ROOT / "app" / "app_pages" / "overview.py"),
                title="Vue d’ensemble",
                icon=":material/dashboard:",
                help="Consultez le modèle retenu, ses métriques réelles, les derniers essais MLflow et l’état des services.",
            ),
            st.Page(
                str(PROJECT_ROOT / "app" / "app_pages" / "data_explorer.py"),
                title="Exploration des données",
                icon=":material/table_chart:",
                help="Explorez le jeu de données réel, la répartition des résiliations, les valeurs manquantes et les variables.",
            ),
            st.Page(
                str(PROJECT_ROOT / "app" / "app_pages" / "predictions.py"),
                title="Prédictions",
                icon=":material/online_prediction:",
                help="Soumettez un profil client à l’API FastAPI et consultez son risque de résiliation calculé en direct.",
            ),
        ],
        "Atelier de modèles": [
            st.Page(
                str(PROJECT_ROOT / "app" / "app_pages" / "experiments.py"),
                title="Expériences AutoML",
                icon=":material/science:",
                help="Comparez les modèles candidats à partir des métriques MLflow et consultez l’évaluation finale.",
            ),
            st.Page(
                str(PROJECT_ROOT / "app" / "app_pages" / "model_registry.py"),
                title="Registre des modèles",
                icon=":material/inventory_2:",
                help="Consultez les versions enregistrées, leur statut, leurs métriques et le modèle servi par l’API.",
            ),
        ],
        "Exploitation": [
            st.Page(
                str(PROJECT_ROOT / "app" / "app_pages" / "monitoring.py"),
                title="Supervision",
                icon=":material/monitoring:",
                help="Vérifiez l’API et Prometheus et suivez le volume des requêtes, leur débit et leur latence.",
            ),
        ],
    },
    position="sidebar",
)

pages.run()
