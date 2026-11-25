# Tests et preuves locales

État vérifié le 25 novembre 2026. Les contrôles ci-dessous portent sur les
fichiers et services présents sur la machine ; ils ne prouvent pas l’état d’un
service distant ni d’une exécution GitHub Actions.

## Tests automatisés

Commande complète après les corrections d’affichage :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Dernière exécution complète effectuée le 25 novembre 2026 après l’ajout de la préparation,
l’optimisation Optuna et le correctif des noms de variables : **18 réussis,
0 échoué, 1 avertissement**. L’avertissement `StarletteDeprecationWarning`
concerne l’usage de `httpx` avec `starlette.testclient`. Cette exécution valide
le code local; elle ne valide pas que les conteneurs en cours servent les
derniers artefacts.

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

Le profil de contrôle identique a renvoyé 0,3468 avec le modèle local mis à
jour et 0,294 avec l’API du conteneur encore actif. Les deux réponses indiquent
un risque faible, mais cette différence confirme que le conteneur API ne sert
pas le dernier artefact. Ces valeurs ne sont pas des KPI généraux.

## Streamlit

- L’image du dashboard a été reconstruite le 23 novembre 2026 et le conteneur
  sert la version française; il n’a pas été reconstruit après le dernier
  entraînement Optuna.
- Les six liens de navigation ont été activés dans le navigateur. Les six URL
  et titres de page attendus ont été observés, sans réponse HTTP 4xx/5xx pendant
  cette navigation.
- Un accès direct à une URL de sous-page a aussi produit des 404 sur les sondes
  internes `/{page}/_stcore/health` et `/{page}/_stcore/host-config`. Les pages
  et liens de navigation eux-mêmes ne sont pas en 404 ; les erreurs concernent
  ces requêtes relatives internes de Streamlit.
- Le formulaire a envoyé un profil réel à FastAPI : **64,0 %**, risque élevé,
  seuil 35 %, modèle GradientBoosting. Ce résultat provient de la version de
  service capturée avant le dernier entraînement.
- Le 25 novembre, les endpoints `localhost:8501/`, `localhost:8000/`,
  `/health`, `/metrics`, `localhost:9090/-/healthy` et
  `localhost:3000/api/health` ont tous répondu HTTP 200. Une nouvelle
  prédiction API a répondu HTTP 200 avec une probabilité de 0,294.
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
- Les quatre conteneurs étaient `Up` après la reconstruction ciblée du
  dashboard : API, Streamlit, Prometheus et Grafana.
- Lors du contrôle du 24 novembre, les ports hôtes 8000, 8501, 9090 et 3000
  étaient publiés et les endpoints de santé accessibles.
- Aucun build n’a été effectué après le dernier entraînement. Lors du dernier
  contrôle d’espace disponible, C: avait environ 6,45 Go libres et les images
  API/dashboard occupaient environ 1,16 Go chacune. Aucun nettoyage,
  suppression d’image, suppression de volume ou de donnée n’a été effectué.
- Prometheus `/-/healthy` : HTTP 200.
- Le target `automl-api` : `health=up`, et la requête Prometheus `up{job="automl-api"}` renvoie `1`.
- Grafana `/api/health` : HTTP 200 et base `ok`; la capture authentifiée
  `grafana-dashboard.png` montre la disponibilité API (`Last: 1`) et le débit
  réel de requêtes.
- La source Prometheus et le dashboard JSON sont provisionnés localement.

## CI, diff et secrets

- Le workflow `.github/workflows/ci.yml` installe `requirements.txt` puis lance
  `pytest tests/`. Le workflow distant n’a pas été déclenché ni consulté.
- `git diff --check` : OK. Git a seulement signalé la conversion habituelle
  LF/CRLF pour le notebook 02 et `tests/test_api.py`.
- La recherche de marqueurs de secrets dans les fichiers suivis n’a rien
  détecté ; aucun fichier `.env*` n’a été trouvé à la racine.

## Captures

Les captures d’écran des six pages, du résultat de prédiction, de Swagger,
des cibles Prometheus, du dashboard Grafana, de MLflow Tracking et du registre
MLflow sont disponibles dans [docs/screenshots/](screenshots/README.md).
L’architecture, le prétraitement, l’évaluation finale, Pytest et les
conteneurs restent à capturer depuis des vues d’exécution authentiques. Les six
captures historiques sous `screenshots/tests/` sont conservées.

## Avertissements de l’entraînement

L’exécution Optuna a également affiché l’avertissement scikit-learn concernant
la dépréciation future de `SVC(probability=True)` et l’avertissement MLflow sur
la désérialisation d’artefacts sérialisés. Ils n’ont pas été masqués et ne font
pas échouer l’entraînement. MLflow UI fonctionne localement; MLflow n’est pas
un service Docker Compose.
