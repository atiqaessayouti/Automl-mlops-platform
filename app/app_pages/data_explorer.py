import pandas as pd
import streamlit as st

from app.data_access import load_churn_dataset
from app.ui import format_count, format_percent, metric_card, page_header


FEATURE_LABELS = {
    "gender": "Genre",
    "SeniorCitizen": "Client senior",
    "Partner": "Partenaire",
    "Dependents": "Personnes à charge",
    "tenure": "Ancienneté (mois)",
    "PhoneService": "Téléphonie",
    "MultipleLines": "Lignes multiples",
    "InternetService": "Service Internet",
    "OnlineSecurity": "Sécurité en ligne",
    "OnlineBackup": "Sauvegarde en ligne",
    "DeviceProtection": "Protection des appareils",
    "TechSupport": "Assistance technique",
    "StreamingTV": "Télévision en streaming",
    "StreamingMovies": "Films en streaming",
    "Contract": "Type de contrat",
    "PaperlessBilling": "Facture dématérialisée",
    "PaymentMethod": "Mode de paiement",
    "MonthlyCharges": "Frais mensuels",
    "TotalCharges": "Frais cumulés",
    "Churn": "Résiliation",
}
VALUE_LABELS = {
    "Yes": "Oui",
    "No": "Non",
    "Female": "Femme",
    "Male": "Homme",
    "No phone service": "Pas de service téléphonique",
    "No internet service": "Pas de service Internet",
    "Month-to-month": "Mensuel",
    "One year": "Un an",
    "Two year": "Deux ans",
    "Bank transfer (automatic)": "Virement automatique",
    "Credit card (automatic)": "Carte bancaire automatique",
    "Electronic check": "Chèque électronique",
    "Mailed check": "Chèque postal",
}


page_header(
    "Exploration des données",
    "Explorez le jeu de données réel sur les résiliations télécom. Les identifiants clients sont exclus de l’aperçu.",
    ":material/table_chart:",
)

dataset, dataset_error = load_churn_dataset()
if dataset is None:
    st.warning(dataset_error or "Le jeu de données source est indisponible.")
    st.stop()

display_data = dataset.drop(columns=["customerID"], errors="ignore").rename(columns=FEATURE_LABELS).copy()
for column in display_data.select_dtypes(include=["object", "str"]):
    display_data[column] = display_data[column].replace(VALUE_LABELS)
target_column = FEATURE_LABELS["Churn"]
missing_total = int(display_data.isna().sum().sum())
numeric_count = len(display_data.select_dtypes(include="number").columns)
categorical_count = len(display_data.columns) - numeric_count

with st.grid(4, min_column_width=180, gap="medium"):
    metric_card("Clients", format_count(len(display_data)), "Lignes du jeu de données source", "data-records", "database")
    metric_card("Variables", f"{len(display_data.columns) - 1}", "Hors variable cible", "data-features", "view_column")
    metric_card("Valeurs manquantes", format_count(missing_total), "Dans les données affichées", "data-missing", "rule")
    metric_card(
        "Taux de résiliation",
        format_percent(display_data[target_column].eq("Oui").mean())
        if target_column in display_data
        else "—",
        "Part observée de clients résiliés",
        "data-churn-rate",
        "monitoring",
    )

st.subheader("Répartition des résiliations", anchor=False)
if target_column not in display_data:
    st.warning(f"La variable cible attendue « {target_column} » est absente du jeu de données.")
else:
    target_counts = (
        display_data[target_column]
        .value_counts(dropna=False)
        .rename_axis("Résiliation")
        .reset_index(name="Clients")
    )
    target_col, summary_col = st.columns([1.2, 1], gap="large")
    with target_col:
        st.bar_chart(
            target_counts,
            x="Résiliation",
            y="Clients",
            color="Résiliation",
            alt="Nombre réel de clients ayant résilié ou non leur abonnement",
        )
    with summary_col:
        with st.container(border=True, key="panel-data-profile"):
            st.markdown("#### Profil des variables")
            st.metric("Variables numériques", numeric_count)
            st.metric("Variables catégorielles", categorical_count)
            st.caption("Les décomptes viennent du fichier source ; aucune donnée fictive n’est ajoutée.")

st.subheader("Valeurs manquantes par variable", anchor=False)
missing_by_column = (
    display_data.isna()
    .sum()
    .rename("Valeurs manquantes")
    .loc[lambda values: values.gt(0)]
    .sort_values(ascending=False)
)
if missing_by_column.empty:
    st.success("Aucune valeur manquante n’a été détectée dans le jeu de données source.")
else:
    st.bar_chart(
        missing_by_column,
        alt="Nombre de valeurs manquantes observées dans chaque variable concernée",
    )
    st.dataframe(
        missing_by_column.rename_axis("Variable").reset_index(),
        hide_index=True,
        alt="Variables du jeu de données source contenant des valeurs manquantes",
    )

st.subheader("Distribution d’une variable", anchor=False)
feature_options = [column for column in display_data.columns if column != target_column]
selected_feature = st.selectbox("Choisir une variable", feature_options, key="data-feature-select")
feature = display_data[selected_feature]
if pd.api.types.is_numeric_dtype(feature):
    observed = feature.dropna()
    if observed.empty:
        st.info(f"Aucune valeur observée n’est disponible pour « {selected_feature} ».")
    else:
        binned = pd.cut(observed, bins=min(12, max(2, observed.nunique()))).value_counts(sort=False)
        distribution = binned.rename_axis("Intervalle").reset_index(name="Clients")
        distribution["Intervalle"] = distribution["Intervalle"].astype(str)
        st.bar_chart(
            distribution,
            x="Intervalle",
            y="Clients",
            alt=f"Valeurs observées pour « {selected_feature} », regroupées par intervalles",
        )
else:
    distribution = (
        feature.value_counts(dropna=False)
        .head(15)
        .rename_axis("Modalité")
        .reset_index(name="Clients")
    )
    distribution["Modalité"] = distribution["Modalité"].fillna("Manquant").astype(str)
    st.bar_chart(
        distribution,
        x="Modalité",
        y="Clients",
        alt=f"Nombre de clients par modalité pour « {selected_feature} »",
    )

st.subheader("Aperçu du jeu de données", anchor=False)
st.dataframe(
    display_data.head(25),
    hide_index=True,
    alt="25 premières lignes des données télécom, sans identifiant client",
)
