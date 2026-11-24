# Architecture de la plateforme

## Vue générale

```mermaid
flowchart LR
    RAW[data/raw/Telco-Customer-Churn.csv] --> NOTEBOOK[Notebook 02 : nettoyage et prétraitement]
    NOTEBOOK --> PROCESSED[data/processed : train/test et cible]
    NOTEBOOK --> PREPROCESSOR[models/preprocessor.joblib]
    PROCESSED --> DVC[DVC : données et stage train_and_register]
    PREPROCESSOR --> TRAIN[src/train.py]
    DVC --> TRAIN
    TRAIN --> MLFLOW[mlflow.db et mlruns]
    TRAIN --> FINAL[models/churn_pipeline.joblib]
    FINAL --> API[FastAPI : 8000]
    API --> STREAMLIT[Streamlit : 8501]
    API --> PROMETHEUS[Prometheus : 9090]
    PROMETHEUS --> GRAFANA[Grafana : 3000]
    GHA[GitHub Actions] --> PYTEST[Pytest]
```

## Composants

| Composant | Rôle | Preuve locale |
|---|---|---|
| Notebook 02 | Nettoyage, séparation stratifiée, apprentissage du préprocesseur sur train uniquement, encodage et export CSV/Joblib | `notebooks/02_data_preprocessing.ipynb` |
| DVC | Suivi des données préparées, du préprocesseur et du stage d’entraînement | `dvc.yaml`, `dvc.lock`, `data/processed.dvc`, `models/preprocessor.joblib.dvc` |
| Entraînement | Comparaison de quatre classifieurs, sélection par F1 de validation, évaluation finale sur le test | `src/train.py`, `reports/metrics.json` |
| MLflow | Runs, paramètres, métriques, modèles enregistrés et versions | `mlflow.db`, `mlruns/` (locaux et ignorés par Git) |
| API | Chargement du pipeline Joblib, prédiction et métriques HTTP | `src/api/main.py`, `src/predict.py` |
| Interface | Six pages Streamlit en français dans la source actuelle | `app/streamlit_app.py`, `app/app_pages/` |
| Monitoring | Scrape Prometheus de FastAPI et dashboard provisionné Grafana | `prometheus.yml`, `monitoring/grafana/` |
| CI | Installation des dépendances et exécution Pytest sur push/PR | `.github/workflows/ci.yml` |

## Flux de prédiction

1. Streamlit construit un profil client avec les modalités attendues par le
   modèle.
2. Le dashboard envoie le profil à `POST /predict`.
3. FastAPI valide le corps JSON, transforme le profil avec le préprocesseur
   sauvegardé, puis calcule la probabilité avec le modèle servi.
4. L’API renvoie le modèle, la probabilité, le seuil, la décision et le niveau
   de risque.
5. L’instrumentateur FastAPI expose les mesures à `/metrics`.
6. Prometheus scrappe `api:8000` toutes les cinq secondes ; Grafana interroge
   Prometheus.

## Portée et limites

- Le stage DVC courant commence à l’entraînement ; le prétraitement notebook
  n’est pas automatisé par `dvc.yaml`.
- L’étude Optuna décrite dans le notebook 06 est indépendante du stage de
  production.
- La base MLflow et ses artefacts sont locaux ; aucun serveur ou stockage
  distant n’est configuré.
- Le fichier de suivi DVC est à jour localement, mais aucun remote n’existe
  pour restaurer les fichiers absents sur une autre machine.
- `docker-compose.yml` utilise `depends_on` avec `service_started`, sans
  healthchecks Docker dédiés.
- Le conteneur Streamlit a été reconstruit pour servir la source française ;
  les quatre services Compose étaient actifs lors du contrôle.
- Les sondes Prometheus sont visibles dans l’interface Supervision. Le rendu
  du dashboard Grafana n’a pas pu être vérifié sans accès authentifié.
