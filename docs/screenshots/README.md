# Captures d’écran de soutenance

Les captures ont été réalisées sur les services locaux réellement exécutés.
Les six captures de pages proviennent du dashboard français déployé sur
`localhost:8501` après son redéploiement du 23 novembre 2026. Aucune maquette ni
capture simulée n’est utilisée.

## Captures disponibles

| Fichier | Contenu | Dimensions |
|---|---|---:|
| `overview.png` | Vue d’ensemble, KPI réels et état de la plateforme | 1920 × 1080 |
| `data-explorer.png` | Exploration du jeu de données réel | 1920 × 1080 |
| `experiments.png` | Comparaison des essais et métriques MLflow | 1920 × 1080 |
| `prediction-form.png` | Formulaire de prédiction | 1920 × 1080 |
| `prediction-result.png` | Résultat réel : 64,0 %, risque élevé, GradientBoosting | 571 × 431 |
| `model-registry.png` | Registre affiché dans Streamlit | 1920 × 1080 |
| `monitoring.png` | État des services affiché dans Streamlit | 1920 × 1080 |
| `fastapi-swagger.png` | Documentation OpenAPI FastAPI | 1920 × 1080 |
| `prometheus-targets.png` | Cible `automl-api` active dans Prometheus | 1920 × 1080 |

`prediction-result-card.png` est une copie de la capture du résultat.
`prediction-result-verified.png` est une capture de viewport incomplète ; elle
ne constitue pas une preuve du résultat et ne doit pas être utilisée pour la
soutenance.

## Inventaire demandé

| # | Capture attendue | État |
|---:|---|---|
| 1 | Architecture générale | À capturer |
| 2 | Vue d’ensemble Streamlit | Capturée : `overview.png` |
| 3 | Exploration des données | Capturée : `data-explorer.png` |
| 4 | Prétraitement | À capturer depuis un notebook réellement exécuté |
| 5 | Expérimentations AutoML | Capturée : `experiments.png` |
| 6 | Comparaison des modèles | Capturée : page Expériences, métriques réelles |
| 7 | Résultats finaux d’évaluation | À capturer séparément depuis `reports/metrics.json` |
| 8 | MLflow Tracking | Non disponible en interface : aucun serveur MLflow n’est déployé |
| 9 | Model Registry | Capturée dans Streamlit : `model-registry.png` ; interface MLflow non déployée |
| 10 | Formulaire de prédiction | Capturé : `prediction-form.png` |
| 11 | Résultat d’une prédiction réelle | Capturé : `prediction-result.png` |
| 12 | Monitoring Streamlit | Partielle : états de service capturés ; graphiques hors viewport |
| 13 | Cibles Prometheus | Capturée : `prometheus-targets.png` |
| 14 | Dashboard Grafana | Non vérifiée ; authentification requise, aucun identifiant fourni |
| 15 | Documentation Swagger FastAPI | Capturée : `fastapi-swagger.png` |
| 16 | Résultat Pytest | À capturer depuis une exécution réelle |
| 17 | Conteneurs Docker actifs | À capturer |
| 18 | GitHub Actions | Bloqué jusqu’à l’autorisation explicite de publication |

Les six captures historiques de `screenshots/tests/` sont conservées et ne
représentent pas la version finale du dashboard.
