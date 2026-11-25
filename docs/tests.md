# Tests et preuves locales

État vérifié le 25 novembre 2026. Les contrôles ci-dessous portent sur les
fichiers et services présents sur la machine ; ils ne prouvent pas l’état d’un
service distant ni d’une exécution GitHub Actions.

## Tests automatisés

Commande complète après les corrections d’affichage :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Dernière exécution complète effectuée le 25 novembre 2026 après le correctif
d’intégration MLflow/dashboard : **20 réussis, 0 échoué, 1 avertissement**.
L’avertissement `StarletteDeprecationWarning` concerne l’usage de `httpx` avec
`starlette.testclient`. Un premier lancement en parallèle des vérifications
des services a dépassé le délai interne de 3 secondes d’un test AppTest;
relancé seul, ce test a réussi, puis la suite complète relancée séquentiellement
a réussi. L’alignement des artefacts en conteneur a été vérifié séparément par
comparaison de leurs empreintes SHA-256 avec les fichiers locaux.

## Données, modèles et DVC

| Contrôle | Résultat observé |
|---|---|
| Dataset brut | Présent, 7 043 lignes × 21 colonnes |
| Train prétraité | 5 634 × 45, non vide, aucune valeur manquante |
| Test prétraité | 1 409 × 45, non vide, aucune valeur manquante |
| Cible train/test | Valeurs uniquement `0` et `1` |
| Modèle final et préprocesseur | Présents localement |
| `dvc status` | « Data and pipelines are up to date. » |
| `dvc repro` | Succès ; 40 essais Optuna complétés (10 par modèle) |
| Remote DVC | Absent ; `dvc status --cloud` échoue car aucun remote n’est défini |
| Modèle local | GradientBoosting ; accuracy 0,7807, precision 0,5678, recall 0,7273, F1 0,6377, ROC-AUC 0,8454 |
| Sélection | F1 validation 0,6537 au seuil 0,35 ; jeu test réservé à l’évaluation finale |

La génération de prétraitement a aussi été exécutée dans un répertoire
temporaire : les exports correspondaient aux CSV préparés existants; le
préprocesseur est ajusté uniquement sur le train.

## FastAPI

Contrôle HTTP local sur `http://127.0.0.1:8000` :

| Requête | Résultat |
|---|---|
| `GET /` | HTTP 200 |
| `GET /health` | HTTP 200 ; modèle `GradientBoosting`, seuil 0,35 |
| `GET /docs` | HTTP 200 |
| `GET /metrics` | HTTP 200 ; métriques Prometheus présentes |
| `POST /predict` avec un profil client valide | HTTP 200 ; réponse du modèle réel |
| `POST /predict` avec `tenure="not-a-number"` | HTTP 422, validation Pydantic |

Avant reconstruction, le profil de contrôle identique renvoyait 0,294 dans
l’ancien conteneur et 0,3468 avec le nouvel artefact local. Après reconstruction
ciblée, l’API conteneurisée et le modèle local renvoient tous deux 0,3468
(`Low Risk (Stay)`). Ces valeurs ne sont pas des KPI généraux.

## Streamlit

- Les images API et dashboard ont été reconstruites après le dernier
  entraînement Optuna; le dashboard a été reconstruit une seconde fois pour
  normaliser les noms des métriques MLflow consommées par les pages.
- Les six liens de navigation ont été activés dans le navigateur. Les six URL
  et titres de page attendus ont été observés, sans réponse HTTP 4xx/5xx pendant
  cette navigation.
- Un accès direct à une URL de sous-page a aussi produit des 404 sur les sondes
  internes `/{page}/_stcore/health` et `/{page}/_stcore/host-config`. Les pages
  et liens de navigation eux-mêmes ne sont pas en 404 ; les erreurs concernent
  ces requêtes relatives internes de Streamlit.
- Le formulaire a envoyé un profil réel à FastAPI : **67,9 %**, risque élevé,
  seuil 35 %, modèle GradientBoosting. Cette entrée est différente du profil de
  contrôle qui donne 0,3468.
- Le 25 novembre, après reconstruction, les endpoints `localhost:8501/`, `localhost:8000/`,
  `/health`, `/metrics`, `localhost:9090/-/healthy` et
  `localhost:3000/api/health` ont tous répondu HTTP 200. Une nouvelle
  prédiction API de contrôle a répondu HTTP 200 avec une probabilité de 0,3468.
- Les graphiques sont rendus avec des séries réelles. Des avertissements Vega
  non bloquants persistent : liaison d’échelle sur des axes catégoriels,
  canal `yOffset` vide, et étendues infinies transitoires au premier rendu des
  graphiques de supervision. Après chargement, les courbes Prometheus affichent
  des points et des axes finis. Les tests vérifient que F1/ROC-AUC ne sont pas
  empilés et que les points non finis sont écartés.
- L’avertissement `Pandas4Warning` sur `select_dtypes` a été corrigé en
  sélectionnant explicitement les types `object` et `str`. Après reconstruction,
  les logs du dashboard ne contiennent plus cet avertissement.

## Docker, Prometheus et Grafana

- `docker compose config --quiet` s’est terminé correctement.
- Les quatre conteneurs sont `Up` après les reconstructions ciblées de l’API et
  du dashboard : API, Streamlit, Prometheus et Grafana.
- Le 25 novembre, après les reconstructions ciblées, les ports hôtes 8000,
  8501, 9090 et 3000 étaient publiés et les endpoints de santé accessibles.
- Les builds ciblés `api dashboard`, puis `dashboard` (pour inclure le correctif
  MLflow) ont réussi avec le cache. Aucun nettoyage, suppression d’image,
  suppression de volume ou de donnée n’a été effectué.
- Prometheus `/-/healthy` : HTTP 200.
- Le target `automl-api` : `health=up`, et la requête Prometheus `up{job="automl-api"}` renvoie `1`.
- Grafana `/api/health` : HTTP 200 et base `ok`; la capture authentifiée
  `grafana-dashboard.png` montre la disponibilité API (`Last: 1`) et le débit
  réel de requêtes.
- La source Prometheus et le dashboard JSON sont provisionnés localement.

## CI, diff et secrets

- Le workflow `.github/workflows/ci.yml` installe `requirements.txt` puis lance
  `pytest tests/`. Le run GitHub Actions #6 sur `main`, commit `736045f`, est
  terminé avec succès; le job `test` et l’étape Pytest sont réussis :
  [résultat du workflow](https://github.com/atiqaessayouti/Automl-mlops-platform/actions/runs/45530432720).
- `git diff --check` : OK. Git a seulement signalé la conversion habituelle
  LF/CRLF pour le notebook 02 et `tests/test_api.py`.
- La recherche de marqueurs de secrets dans les fichiers suivis n’a rien
  détecté ; aucun fichier `.env*` n’a été trouvé à la racine.

## Captures

Les captures d’écran des six pages, du résultat de prédiction, de Swagger,
des cibles Prometheus, du dashboard Grafana, de MLflow Tracking et du registre
MLflow sont disponibles dans [docs/screenshots/](screenshots/README.md).
L’évaluation finale est également capturée. L’architecture, le prétraitement,
Pytest et les conteneurs restent à capturer depuis des vues d’exécution
authentiques. Les six captures historiques sous `screenshots/tests/` sont
conservées.

## Avertissements de l’entraînement

L’exécution Optuna a également affiché l’avertissement scikit-learn concernant
la dépréciation future de `SVC(probability=True)` et l’avertissement MLflow sur
la désérialisation d’artefacts sérialisés. Ils n’ont pas été masqués et ne font
pas échouer l’entraînement. MLflow UI fonctionne localement; MLflow n’est pas
un service Docker Compose.
