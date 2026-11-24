# Rapport de conformité local

- **Projet :** plateforme AutoML/MLOps de prédiction du churn télécom
- **Référence :** `Document_de_cadrage_AutoML_MLOps (1).pdf`
- **État du contrôle :** 23 novembre 2026
- **Branche observée :** `feature/devops-deployment`
**Périmètre :** audit et validation locale. Aucun commit, push, changement de
branche, opération GitHub, suppression de données ou de volume. Seul le service
dashboard a été reconstruit après autorisation.

## Tableau de conformité

| Domaine / exigence | État | Preuves et résultats |
|---|---|---|
| Dataset Telco et exploration | Terminé et vérifié | CSV brut présent : 7 043 × 21 ; notebook 01 et pages d’exploration disponibles |
| Prétraitement et séparation | Terminé et vérifié | Notebook 02 ; séparation stratifiée 80/20 ; imputeurs/scaler/encodeur ajustés sur train ; artefacts vérifiés (5 634/1 409 × 45) |
| Validation dédiée du schéma des données | Partiellement terminé | Le notebook contrôle les valeurs manquantes, doublons et formes ; aucune validation automatisée de schéma n’est intégrée au pipeline DVC |
| Entraînement et comparaison AutoML | Terminé et vérifié | `src/train.py` compare quatre modèles, choisit sur F1 de validation, réentraîne sur train et évalue sur test ; `reports/metrics.json` cohérent |
| Optimisation d’hyperparamètres | Partiellement terminé | Notebooks 03, 05 et 06 documentent GridSearchCV, RandomizedSearchCV et Optuna ; le stage d’entraînement actif ne les exécute pas et l’environnement standard ne dépend pas d’Optuna |
| Algorithme XGBoost envisagé | Non réalisé | Les quatre modèles actifs sont scikit-learn ; XGBoost n’est pas installé ni comparé dans le pipeline courant |
| DVC et reproductibilité locale | Partiellement terminé | `dvc status` et `dvc repro` locaux : à jour ; le prétraitement notebook n’est pas un stage DVC |
| Récupération DVC à distance | Impossible à vérifier | Aucun remote configuré ; `dvc status --cloud` échoue avec « no remote specified ». Les artefacts actuellement absents ne sont pas récupérables à distance |
| MLflow Tracking et Registry | Partiellement terminé | SQLite locale : 10 runs, quatre candidats, modèle `TelcoChurnModel`, deux versions `READY`. Aucun serveur MLflow distant ni persistance Git vérifiés |
| FastAPI et vraies prédictions | Terminé et vérifié | `/`, `/health`, `/docs`, `/metrics` HTTP 200 ; vraie prédiction HTTP 200 ; entrée `tenure` invalide HTTP 422 |
| Streamlit — six pages et API | Partiellement terminé | Image française reconstruite ; six pages ouvertes par la navigation sans échec HTTP ; prédiction réelle via le formulaire (64,0 %, risque élevé). Avertissements Vega non bloquants et 404 observées uniquement sur deux sondes internes après accès direct |
| Docker Compose et quatre services | Partiellement terminé | `docker compose config --quiet` valide ; les quatre conteneurs `Up`, dashboard reconstruit seul. Healthchecks non configurés et `depends_on` reste sur `service_started` |
| Prometheus | Terminé et vérifié | `/‑/healthy` HTTP 200 ; target `automl-api` sain ; métrique `up=1` |
| Grafana | Partiellement terminé | `/api/health` HTTP 200 ; source et dashboard provisionnés. L’API de consultation renvoie 401 sans authentification ; rendu du dashboard non confirmé pendant cet audit |
| Pytest local | Terminé et vérifié | 15 tests réussis, aucun échec ; un avertissement Starlette/httpx |
| GitHub Actions / CI-CD | Partiellement terminé | Workflow présent et lance pytest sur push/PR ; aucun run distant vérifié, conformément à l’interdiction de publication |
| Documentation finale | Partiellement terminé | README, installation, architecture, tests et rapport actualisés ; certaines preuves et captures opérationnelles restent à produire |
| Captures de démonstration | Partiellement terminé | Captures réelles des six pages, d’une prédiction, de Swagger et de Prometheus. Captures Grafana, Pytest, Docker, prétraitement et certaines vues d’évaluation restent manquantes |
| Secrets publiables | Terminé et vérifié | Recherche de motifs de clés et secrets dans les fichiers suivis : aucun résultat ; pas de `.env*` à la racine |

## Résultats réels actuellement disponibles

Le rapport local `reports/metrics.json` indique le GradientBoosting au seuil
0,35 : accuracy 0,7764, precision 0,5626, recall 0,7086, F1 0,6272 et ROC-AUC
0,8434. Les données préparées sont non vides, sans valeurs manquantes dans les
matrices encodées, et la cible est `0/1`.

Après la reconstruction ciblée, les quatre conteneurs sont `Up`. L’API, ses
métriques, Prometheus et Grafana health répondent HTTP 200 ; la cible Prometheus
`automl-api` est `up=1`. Une prédiction directe a renvoyé 0,294 (risque faible)
pour son profil. Le formulaire Streamlit a renvoyé 64,0 % (risque élevé) pour
un profil différent. Les six pages françaises sont servies sur 8501.

## Corrections appliquées pendant cet audit

- Les graphiques de comparaison F1/ROC-AUC dans les pages Vue d’ensemble et
  Expériences sont explicitement non empilés (`stack=False`) pour éviter de
  sommer deux métriques distinctes sur le même axe.
- La page Supervision ignore les points de série temporelle Prometheus non
  numériques ou infinis, qui provoquaient des avertissements Vega lorsque les
  données étaient inexploitables.
- L’exploration sélectionne explicitement les types `object` et `str` pour
  éviter `Pandas4Warning` ; les logs après reconstruction ne montrent plus cet
  avertissement.
- L’image du seul service dashboard a été reconstruite avec le cache des
  dépendances, puis le conteneur dashboard recréé. Aucun autre service, volume,
  image ou jeu de données n’a été supprimé.
- Documentation française créée ou complétée dans `README.md` et `docs/`.
- Les six pages ont été vérifiées dans le navigateur ; la capture du résultat
  montre une vraie prédiction et l’API a également été testée directement.

## Fichiers modifiés dans l’état courant

Les changements préexistants au début de cette mission sont conservés. Les
fichiers créés/modifiés pour ce rapport sont :

- `README.md`
- `app/app_pages/overview.py`
- `app/app_pages/experiments.py`
- `docs/architecture.md`
- `docs/installation.md`
- `docs/tests.md`
- `docs/rapport-final.md`
- `docs/screenshots/README.md`
- `docs/screenshots/fastapi-swagger.png`
- `docs/screenshots/prometheus-targets.png`
- `app/app_pages/monitoring.py`
- `app/app_pages/data_explorer.py`
- `tests/test_streamlit_app.py`
- `docs/screenshots/overview.png`
- `docs/screenshots/data-explorer.png`
- `docs/screenshots/experiments.png`
- `docs/screenshots/model-registry.png`
- `docs/screenshots/monitoring.png`
- `docs/screenshots/prediction-form.png`
- `docs/screenshots/prediction-result.png`

Les fichiers et captures préexistants ne sont pas supprimés.

## Tâches restantes avant une validation finale

1. Examiner les avertissements Vega résiduels (`yOffset`, liaison d’échelle,
   étendues infinies transitoires) si une console sans avertissement est exigée.
   Les graphiques s’affichent avec les données réelles après chargement.
2. Vérifier visuellement le dashboard Grafana avec un accès authentifié fourni ;
   aucune authentification n’a été tentée.
3. Décider si Optuna doit être intégré au pipeline DVC actif, plutôt que
   maintenu dans un notebook autonome.
4. Choisir et autoriser un stockage DVC si la récupération distante est
   nécessaire. Aucun remote fictif ne doit être ajouté.
5. Capturer les vues manquantes de prétraitement, résultats finaux, Pytest,
   Docker, Grafana et MLflow Tracking. Aucun serveur MLflow n’est configuré dans
   Compose, et les données de runs ne sont que locales.
6. Ne capturer GitHub Actions qu’après autorisation de publication ; aucun
   commit, push, PR ou run CI distant n’a été effectué.

**Aucune publication GitHub n’a été effectuée.** La phase de publication reste
bloquée jusqu’à une autorisation explicite.
