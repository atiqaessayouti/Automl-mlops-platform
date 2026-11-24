import httpx
import streamlit as st

from app.data_access import API_URL
from app.ui import format_percent, page_header


CHOICE_LABELS = {
    "Female": "Femme",
    "Male": "Homme",
    "Yes": "Oui",
    "No": "Non",
    "No phone service": "Pas de service téléphonique",
    "No internet service": "Pas de service Internet",
    "Month-to-month": "Mensuel",
    "One year": "Un an",
    "Two year": "Deux ans",
    "Electronic check": "Chèque électronique",
    "Mailed check": "Chèque postal",
    "Bank transfer (automatic)": "Virement automatique",
    "Credit card (automatic)": "Carte bancaire automatique",
    "Fiber optic": "Fibre optique",
    "DSL": "ADSL",
}
RISK_LABELS = {
    "High Risk (Churn)": "Risque élevé de résiliation",
    "Low Risk (Stay)": "Risque faible — client susceptible de rester",
}


page_header(
    "Prédictions",
    "Évaluez un profil client avec le modèle FastAPI en service. Chaque résultat provient d’une requête réelle à l’API.",
    ":material/online_prediction:",
)

with st.form("customer-prediction-form", border=False):
    profile, billing, services = st.columns(3, gap="medium")
    with profile:
        with st.container(border=True, key="panel-customer-profile"):
            st.subheader("Profil du client", anchor=False)
            gender = st.selectbox(
                "Genre",
                ["Female", "Male"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-gender",
            )
            senior = st.selectbox(
                "Client senior",
                [0, 1],
                format_func=lambda value: "Oui" if value == 1 else "Non",
                key="prediction-senior",
            )
            partner = st.selectbox(
                "Partenaire",
                ["No", "Yes"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-partner",
            )
            dependents = st.selectbox(
                "Personnes à charge",
                ["No", "Yes"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-dependents",
            )
            tenure = st.slider("Ancienneté (mois)", 0, 72, 2, key="prediction-tenure")
    with billing:
        with st.container(border=True, key="panel-plan-billing"):
            st.subheader("Abonnement et facturation", anchor=False)
            contract = st.selectbox(
                "Type de contrat",
                ["Month-to-month", "One year", "Two year"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-contract",
            )
            paperless = st.selectbox(
                "Facture dématérialisée",
                ["Yes", "No"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-paperless",
            )
            payment = st.selectbox(
                "Mode de paiement",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-payment",
            )
            monthly_charges = st.number_input(
                "Frais mensuels ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.70,
                key="prediction-monthly-charges",
            )
            total_charges = st.number_input(
                "Frais cumulés ($)",
                min_value=0.0,
                max_value=10000.0,
                value=151.65,
                key="prediction-total-charges",
            )
    with services:
        with st.container(border=True, key="panel-services"):
            st.subheader("Services souscrits", anchor=False)
            phone = st.selectbox(
                "Téléphonie",
                ["Yes", "No"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-phone",
            )
            multiple_lines = st.selectbox(
                "Lignes multiples",
                ["No", "Yes", "No phone service"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-multiple-lines",
            )
            internet = st.selectbox(
                "Service Internet",
                ["Fiber optic", "DSL", "No"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-internet",
            )
            online_security = st.selectbox(
                "Sécurité en ligne",
                ["No", "Yes", "No internet service"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-online-security",
            )
            online_backup = st.selectbox(
                "Sauvegarde en ligne",
                ["No", "Yes", "No internet service"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-online-backup",
            )
            device_protection = st.selectbox(
                "Protection des appareils",
                ["No", "Yes", "No internet service"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-device-protection",
            )
            tech_support = st.selectbox(
                "Assistance technique",
                ["No", "Yes", "No internet service"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-tech-support",
            )
            streaming_tv = st.selectbox(
                "Télévision en streaming",
                ["No", "Yes", "No internet service"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-streaming-tv",
            )
            streaming_movies = st.selectbox(
                "Films en streaming",
                ["No", "Yes", "No internet service"],
                format_func=lambda value: CHOICE_LABELS[value],
                key="prediction-streaming-movies",
            )

    submitted = st.form_submit_button(
        "Évaluer le risque de résiliation",
        type="primary",
        icon=":material/bolt:",
        width="stretch",
    )

if submitted:
    client_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    try:
        response = httpx.post(f"{API_URL}/predict", json=client_data, timeout=10.0)
        response.raise_for_status()
        result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"La requête de prédiction vers FastAPI a échoué : {exc}", icon=":material/error:")
    else:
        required_fields = {
            "model",
            "churn_probability",
            "threshold",
            "churn_prediction",
            "risk_label",
        }
        if not isinstance(result, dict) or not required_fields.issubset(result):
            st.error("FastAPI a renvoyé une réponse de prédiction inattendue.")
        else:
            probability = float(result["churn_probability"])
            threshold = float(result["threshold"])
            with st.container(border=True, key="prediction-result"):
                st.caption(f"Réponse en direct de {API_URL}")
                st.subheader("Résultat de la prédiction", anchor=False)
                result_columns = st.columns([1, 2], vertical_alignment="center")
                with result_columns[0]:
                    st.metric("Probabilité de résiliation", format_percent(probability))
                    st.progress(min(max(probability, 0.0), 1.0))
                with result_columns[1]:
                    is_churn = bool(result["churn_prediction"])
                    st.markdown(f"### {RISK_LABELS.get(result['risk_label'], result['risk_label'])}")
                    st.caption(
                        f"Seuil de décision : {format_percent(threshold, 0)} · Modèle : {result['model']}"
                    )
                    if is_churn:
                        st.warning(
                            "Ce profil dépasse le seuil de décision du modèle. "
                            "Une action de fidélisation peut être envisagée.",
                            icon=":material/priority_high:",
                        )
                    else:
                        st.success(
                            "Ce profil se situe sous le seuil de décision du modèle.",
                            icon=":material/check_circle:",
                        )
