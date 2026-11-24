# Installation et exécution

## Prérequis

- Windows 10/11 ou environnement compatible.
- Python 3.11 pour l’environnement local et les tests.
- Docker Desktop avec Docker Compose v2 pour le déploiement.
- Git pour obtenir le code.
- Espace disque suffisant pour Docker : l’image dashboard existante pèse
  environ 1,16 Go. Mesurer l’espace libre avant chaque build.

## Environnement Python local

Depuis la racine du dépôt :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le notebook `06_mlflow_optuna.ipynb` contient sa propre cellule d’installation
d’Optuna et MLflow. L’environnement standard et le pipeline DVC ne garantissent
pas actuellement que cette recherche soit intégrée ou exécutée.

## Préparation et entraînement

1. Vérifier la présence de `data/raw/Telco-Customer-Churn.csv`.
2. Exécuter `notebooks/02_data_preprocessing.ipynb` depuis la racine du projet
   ou depuis `notebooks/`.
3. Vérifier les fichiers train/test et le préprocesseur générés.
4. Exécuter le pipeline disponible :

```powershell
dvc status
dvc repro
```

`dvc repro` vérifie localement le stage d’entraînement. Les artefacts DVC
présents sont à jour sur la machine auditée. Comme aucun remote DVC n’est
configuré, les données suivies qui ne sont pas disponibles localement ne peuvent
pas être restaurées depuis un stockage distant.

## Déploiement Docker Compose

Vérifier la configuration et l’état des services :

```powershell
docker compose config
docker compose ps
```

Démarrer les images disponibles :

```powershell
docker compose up -d
```

L’image dashboard française a été reconstruite et vérifiée localement le
23 novembre 2026. Pour une prochaine mise à jour du seul dashboard :

```powershell
docker compose build dashboard
docker compose up -d --no-deps dashboard
```

Ces commandes ne ciblent pas le service API, les volumes Prometheus/Grafana ou
les données. Elles ne suppriment pas d’images ou de volumes ; ne pas utiliser
`down -v`, `system prune` ou une commande de suppression de volume.

## URLs locales

| Service | URL |
|---|---|
| Streamlit | <http://localhost:8501> |
| FastAPI | <http://localhost:8000> |
| Swagger | <http://localhost:8000/docs> |
| Prometheus | <http://localhost:9090> |
| Grafana | <http://localhost:3000> |

La configuration Grafana présente ne contient pas de secrets. L’API Grafana
nécessite une authentification pour les endpoints `/api/datasources` et
`/api/search` dans l’environnement vérifié.
