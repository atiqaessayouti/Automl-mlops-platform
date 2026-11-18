import json
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import predict_churn

st.set_page_config(page_title="Telco Churn MLOps", page_icon="📊", layout="centered")

st.title("📊 Prédiction du Churn Client (Telco)")
st.caption("Modèle Champion : **SVM** | F1-score : **0.6305** | Seuil optimal : **0.35**")

with st.form("churn_form"):
    st.subheader("Profil du client")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Genre", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
        partner = st.selectbox("Partenaire", ["No", "Yes"])
        dependents = st.selectbox("Personnes à charge", ["No", "Yes"])
        tenure = st.slider("Ancienneté (mois)", 0, 72, 2)
        contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Facturation sans papier", ["Yes", "No"])
        payment = st.selectbox(
            "Moyen de paiement",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Frais mensuels ($)", min_value=0.0, max_value=200.0, value=70.70)
        total_charges = st.number_input("Frais totaux ($)", min_value=0.0, max_value=10000.0, value=151.65)

    with col2:
        phone = st.selectbox("Service téléphonique", ["Yes", "No"])
        multiple_lines = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Service Internet", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Sauvegarde en ligne", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Protection appareil", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Support technique", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Films", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("🔍 Prédire le risque de résiliation", use_container_width=True)

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

    result = predict_churn(client_data)
    prob_pct = result["churn_probability"] * 100

    st.divider()
    st.metric("Probabilité de Churn", f"{prob_pct:.1f} %", delta=f"Seuil : {result['threshold']*100:.0f} %")

    if result["churn_prediction"] == 1:
        st.error(f"⚠️ **{result['risk_label']}** — Action de rétention recommandée.")
    else:
        st.success(f"✅ **{result['risk_label']}** — Client stable.")