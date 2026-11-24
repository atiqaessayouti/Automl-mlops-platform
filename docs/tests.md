# Tests et preuves locales

État vérifié le 23 novembre 2026. Les contrôles ci-dessous portent sur les
fichiers et services présents sur la machine ; ils ne prouvent pas l’état d’un
service distant ni d’une exécution GitHub Actions.

## Tests automatisés

Commande complète après les corrections d’affichage :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Résultat constaté après correction et redéploiement du dashboard :
**15 réussis, 0 échoué** en 41,50 s. Deux tests dédiés
vérifient que les graphiques comparatifs des pages Vue d’ensemble et
Expériences ne superposent pas les scores F1 et ROC-AUC ; un autre vérifie
l’exclusion des points de métriques non finis dans la page Supervision. Un
avertissement est conservé : `StarletteDeprecationWarning` indique que l’usage
de `httpx` avec `starlette.testclient` est déprécié et recommande `httpx2`.

## Données, modèles et DVC

| Contrôle | Résultat observé |
|---|---|
| Dataset brut | Présent, 7 043 lignes × 21 colonnes |
| Train prétraité | 5 634 × 45, non vide, aucune valeur manquante |
| Test prétraité | 1 409 × 45, non vide, aucune valeur manquante |
| Cible train/test | Valeurs uniquement `0` et `1` |
| Modèle final et préprocesseur | Présents localement |
| `dvc status` | « Data and pipelines are up to date. » |
| `dvc repro` | Stage `train_and_register` ignoré car inchangé ; pipeline à jour |
| Remote DVC | Absent ; `dvc status --cloud` échoue car aucun remote n’est défini |

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

La prédiction de contrôle a renvoyé une probabilité de 0,1704 et la décision
`Low Risk (Stay)` pour le profil fourni au test. Ce résultat n’est pas un KPI
général.

## Streamlit

- L’image du dashboard a été reconstruite le 23 novembre 2026 et le conteneur
  recréé ; `localhost:8501` sert maintenant la version française.
- Les six liens de navigation ont été activés dans le navigateur. Les six URL
  et titres de page attendus ont été observés, sans réponse HTTP 4xx/5xx pendant
  cette navigation.
- Un accès direct à une URL de sous-page a aussi produit des 404 sur les sondes
  internes `/{page}/_stcore/health` et `/{page}/_stcore/host-config`. Les pages
  et liens de navigation eux-mêmes ne sont pas en 404 ; les erreurs concernent
  ces requêtes relatives internes de Streamlit.
- Le formulaire a envoyé un profil réel à FastAPI : **64,0 %**, risque élevé,
  seuil 35 %, modèle GradientBoosting. Une requête API de contrôle distincte a
  renvoyé 0,294 (risque faible) pour un autre profil.
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
- `docker compose build dashboard` a utilisé le cache des dépendances ; seul
  `automl-dashboard` a été recréé. Avant le build, C: disposait de
  9 017 081 856 octets libres. Aucun nettoyage, suppression d’image, suppression
  de volume ou de donnée n’a été effectué.
- Prometheus `/-/healthy` : HTTP 200.
- Le target `automl-api` : `health=up`, et la requête Prometheus `up{job="automl-api"}` renvoie `1`.
- Grafana `/api/health` : HTTP 200 et base `ok` (version 13.2.3).
- La source de données Prometheus et le dashboard JSON sont provisionnés par
  les fichiers locaux. Les API Grafana des datasources/recherches renvoient
  HTTP 401 sans authentification ; aucun identifiant n’a été essayé et le
  rendu des graphiques Grafana n’est donc pas confirmé.

## CI, diff et secrets

- Le workflow `.github/workflows/ci.yml` installe `requirements.txt` puis lance
  `pytest tests/`. Le workflow distant n’a pas été déclenché ni consulté.
- `git diff --check` : OK. Git a seulement signalé la conversion habituelle
  LF/CRLF pour le notebook 02 et `tests/test_api.py`.
- La recherche de marqueurs de secrets dans les fichiers suivis n’a rien
  détecté ; aucun fichier `.env*` n’a été trouvé à la racine.

## Captures

Les captures d’écran des six pages, du résultat de prédiction, de Swagger et
des cibles Prometheus sont disponibles dans
[docs/screenshots/](screenshots/README.md). Les vues Grafana, MLflow Tracking,
prétraitement, résultat Pytest et conteneurs restent à capturer. Les six
captures historiques sous `screenshots/tests/` sont conservées.
