# Rapport de conformité local

- **Projet :** plateforme AutoML/MLOps de prédiction du churn télécom
- **Réalisé par :**
  - TIMOURTI SANA
  - ESSAYOUTI ATIQA
  - MISBAH KAOUTAR
- **Référence :** `Document_de_cadrage_AutoML_MLOps (1).pdf`
- **État du contrôle :** 25 novembre 2026, complété par les captures MLflow locales
- **Branche de travail :** `feature/devops-deployment`
- **Branche de livraison :** `main`
- **Commit publié vérifié :** `736045f`
**Périmètre :** audit et validation locale, puis publication sur `origin/main`
après intégration et contrôles finaux. Aucun nettoyage destructif ni suppression
de données ou de volume n’a été effectué.

## Tableau de conformité

| Domaine / exigence | État | Preuves et résultats |
|---|---|---|
| Dataset Telco et exploration | Terminé et vérifié | CSV brut présent : 7 043 × 21 ; notebook 01 et pages d’exploration disponibles |
| Prétraitement et séparation | Terminé et vérifié | `src/preprocess.py` et notebook 02 ; séparation stratifiée 80/20 ; préprocesseur ajusté sur train ; exports comparés (5 634/1 409 × 45) |
| Validation dédiée du schéma des données | Partiellement terminé | Le notebook contrôle les valeurs manquantes, doublons et formes ; aucune validation automatisée de schéma n’est intégrée au pipeline DVC |
| Entraînement et comparaison AutoML | Terminé et vérifié | `src/train.py` compare quatre modèles, choisit sur F1 de validation, réentraîne et évalue sur le jeu test réservé ; métriques vérifiées |
| Optimisation d’hyperparamètres | Terminé et vérifié | Optuna/TPE intégré au stage actif : 10 essais par modèle, validation croisée stratifiée à 3 folds, 40 essais terminés |
| Algorithme XGBoost envisagé | Non réalisé | Les quatre modèles actifs sont scikit-learn ; XGBoost n’est pas installé ni comparé dans le pipeline courant |
| DVC et reproductibilité locale | Partiellement terminé | `dvc status`/`dvc repro` réussis ; prétraitement réexécutable mais manuel avant le stage, pointeurs `.dvc` conservés |
| Récupération DVC à distance | Impossible à vérifier | Aucun remote configuré ; `dvc status --cloud` échoue avec « no remote specified ». Les artefacts actuellement absents ne sont pas récupérables à distance |
| MLflow Tracking et Registry | Partiellement terminé | Base SQLite locale : 15 runs et trois versions `READY`; interface Tracking et registre capturés. Pas de service Compose ni de stockage partagé |
| FastAPI et vraies prédictions | Terminé et vérifié | Endpoints HTTP 200, entrée invalide 422; après reconstruction, la prédiction conteneurisée correspond au modèle local |
| Streamlit — six pages et API | Terminé et vérifié | Six pages rendues; comparaison MLflow corrigée et prédiction réelle du formulaire confirmée après reconstruction |
| Docker Compose et quatre services | Partiellement terminé | Configuration valide et quatre services actifs; API/dashboard mis à jour. Healthchecks Docker non définis et dépendances `service_started` |
| Prometheus | Terminé et vérifié | `/‑/healthy` HTTP 200 ; target `automl-api` sain ; métrique `up=1` |
| Grafana | Terminé et vérifié | `/api/health` HTTP 200; dashboard consulté avec authentification, capture montrant API disponible et requêtes |
| Pytest local | Terminé et vérifié | 20 tests réussis, 0 échec, 1 avertissement Starlette/httpx |
| GitHub Actions / CI-CD | Terminé et vérifié | Run #6 sur `main` (`736045f`) : job `test` et étape Pytest réussis |
| Documentation finale | Partiellement terminé | README, installation, architecture, tests et rapport actualisés; captures authentiques et limitations recensées |
| Captures de démonstration | Partiellement terminé | Captures réelles du dashboard, prédiction, évaluation, Swagger, Prometheus, Grafana, MLflow Tracking et Registry; architecture, prétraitement, Pytest, Docker et CI manquent |
| Secrets publiables | Terminé et vérifié | Recherche de motifs de clés et secrets dans les fichiers suivis : aucun résultat ; pas de `.env*` à la racine |

## Résultats réels actuellement disponibles

Le rapport local `reports/metrics.json` indique le GradientBoosting au seuil
0,35 : accuracy 0,7807, precision 0,5678, recall 0,7273, F1 0,6377 et ROC-AUC
0,8454. Son F1 de validation est 0,6537. Les données préparées sont non vides,
sans valeurs manquantes dans les matrices encodées, et la cible est `0/1`.

Les quatre conteneurs sont `Up`; l’API, Prometheus et Grafana répondent. La
cible `automl-api` vaut `up=1`, et Grafana a été inspecté visuellement. Après
reconstruction ciblée, le même profil de contrôle renvoie 0,3468 à l’API et au
modèle local (risque faible). Le formulaire Streamlit a renvoyé 67,9 % pour un
autre profil (risque élevé).

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
- `src/preprocess.py` rend le prétraitement réexécutable avec ajustement du
  transformateur sur le seul train, puis export des jeux train/test et du
  préprocesseur.
- `src/train.py` intègre Optuna/TPE aux quatre modèles, enregistre les essais,
  sélectionne le meilleur modèle sur le F1 de validation et consigne les
  métriques dans MLflow.
- `src/predict.py` conserve les noms des variables transformées lors de
  l’inférence, supprimant l’avertissement de noms de variables sur la
  prédiction locale.
- Les images API et dashboard ont été reconstruites de façon ciblée après
  l’entraînement Optuna; le dashboard a ensuite été reconstruit pour intégrer
  la normalisation des métriques MLflow. Aucun volume, jeu de données ou image
  utile n’a été supprimé.
- Documentation française créée ou complétée dans `README.md` et `docs/`.
- Les six pages ont été vérifiées dans le navigateur ; la capture du résultat
  montre une vraie prédiction et l’API a également été testée directement.
- Captures authentiques ajoutées pour Grafana et les interfaces MLflow Tracking
  et Registry.
- Normalisation des métriques MLflow `validation_*` dans la couche d’accès aux
  données, afin que le dashboard compare bien les quatre derniers candidats.

## Fichiers inclus dans la livraison locale

- Documentation : `README.md`, `docs/architecture.md`,
  `docs/installation.md`, `docs/tests.md`, `docs/rapport-final.md`,
  `docs/screenshots/README.md`.
- Pipeline et dépendances : `src/preprocess.py`, `src/train.py`,
  `src/predict.py`, `requirements.txt`, `dvc.yaml`, `dvc.lock`.
- Résultats : `reports/metrics.json`, `reports/.gitignore`.
- Tests : `tests/test_preprocess.py`, `tests/test_train.py`,
  `tests/test_predict.py`.
- Captures ajoutées dans ce suivi : `docs/screenshots/grafana-dashboard.png`,
  `docs/screenshots/mlflow-tracking.png`,
  `docs/screenshots/mlflow-registry.png`.

Les autres captures existantes et les changements déjà présents avant cette
séquence sont conservés. Aucun fichier n’a été supprimé.

## Tâches restantes avant une validation finale

1. Capturer les vues manquantes de l’architecture, du prétraitement, de Pytest
   et des conteneurs depuis des exécutions réelles.
2. Le prétraitement reste manuel avant le stage DVC; l’ajouter au pipeline
   nécessite une stratégie compatible avec les pointeurs DVC déjà suivis.
3. Choisir et autoriser un stockage DVC uniquement si la récupération distante
   est nécessaire. Aucun remote fictif ne doit être ajouté.
4. Examiner si une sortie sans avertissement est exigée : Vega au premier
   rendu, Starlette/httpx, `SVC(probability=True)` et désérialisation MLflow
   sont signalés sans masquer les avertissements.
Le commit `736045f` est publié sur `origin/main`; le run GitHub Actions #6 a
terminé avec succès.
