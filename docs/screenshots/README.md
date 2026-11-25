# Captures d’écran de soutenance

Les captures ont été réalisées sur les services locaux réellement exécutés.
Les six captures de pages ont été actualisées depuis le dashboard français
servi sur `localhost:8501` après les reconstructions ciblées du 25 novembre
2026. Aucune maquette ni capture simulée n’est utilisée.

## Captures disponibles

| Fichier | Contenu | Dimensions |
|---|---|---:|
| `overview.png` | Vue d’ensemble, KPI réels et état de la plateforme | 1920 × 1080 |
| `data-explorer.png` | Exploration du jeu de données réel | 1920 × 1080 |
| `experiments.png` | Comparaison des essais et métriques MLflow | 1920 × 1080 |
| `evaluation-results.png` | Résultats finaux d’évaluation depuis la page Expériences | 1920 × 1080 |
| `prediction-form.png` | Formulaire de prédiction | 1920 × 1080 |
| `prediction-result.png` | Résultat réel : 67,9 %, risque élevé, GradientBoosting | 1920 × 1080 |
| `model-registry.png` | Registre affiché dans Streamlit | 1920 × 1080 |
| `monitoring.png` | État des services affiché dans Streamlit; les graphiques sont plus bas dans la page | 1920 × 1080 |
| `fastapi-swagger.png` | Documentation OpenAPI FastAPI | 1920 × 1080 |
| `prometheus-targets.png` | Cible `automl-api` active dans Prometheus | 1920 × 1080 |
| `grafana-dashboard.png` | Dashboard Grafana authentifié : disponibilité API et requêtes | 1920 × 1080 |
| `mlflow-tracking.png` | Runs de l’expérience AutoML dans l’interface MLflow locale | 1920 × 1080 |
| `mlflow-registry.png` | Registre MLflow local et versions enregistrées | 1920 × 1080 |

## Inventaire demandé

| # | Capture attendue | État |
|---:|---|---|
| 1 | Architecture générale | À capturer |
| 2 | Vue d’ensemble Streamlit | Capturée : `overview.png` |
| 3 | Exploration des données | Capturée : `data-explorer.png` |
| 4 | Prétraitement | À capturer depuis un notebook réellement exécuté |
| 5 | Expérimentations AutoML | Capturée : `experiments.png` |
| 6 | Comparaison des modèles | Capturée : page Expériences, métriques réelles |
| 7 | Résultats finaux d’évaluation | Capturés dans `evaluation-results.png` depuis la page Expériences |
| 8 | MLflow Tracking | Capturée : `mlflow-tracking.png` (interface locale, non déployée par Compose) |
| 9 | Model Registry | Capturé dans Streamlit (`model-registry.png`) et MLflow (`mlflow-registry.png`) |
| 10 | Formulaire de prédiction | Capturé : `prediction-form.png` |
| 11 | Résultat d’une prédiction réelle | Capturé : `prediction-result.png` |
| 12 | Monitoring Streamlit | Partielle : états de service capturés ; graphiques hors viewport |
| 13 | Cibles Prometheus | Capturée : `prometheus-targets.png` |
| 14 | Dashboard Grafana | Capturé : `grafana-dashboard.png` après authentification locale |
| 15 | Documentation Swagger FastAPI | Capturée : `fastapi-swagger.png` |
| 16 | Résultat Pytest | À capturer depuis une exécution réelle |
| 17 | Conteneurs Docker actifs | À capturer |
| 18 | GitHub Actions | Workflow #6 réussi; capture d’écran non enregistrée |

Les six captures historiques de `screenshots/tests/` sont conservées et ne
représentent pas la version finale du dashboard. Les captures MLflow montrent
la base locale après le dernier entraînement Optuna; elles ne prouvent pas que
les conteneurs API/dashboard ont été reconstruits avec ces artefacts.
