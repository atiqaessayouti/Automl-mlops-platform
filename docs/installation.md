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

Optuna est déclaré dans `requirements.txt` et utilisé par le stage
`train_and_register` du pipeline DVC.

## Préparation et entraînement

1. Vérifier la présence de `data/raw/Telco-Customer-Churn.csv`.
2. Exécuter le script déterministe de préparation depuis la racine :

```powershell
.\.venv\Scripts\python.exe src\preprocess.py
```

Le notebook `notebooks/02_data_preprocessing.ipynb` reste disponible pour
l’exploration interactive.
3. Vérifier les fichiers train/test et le préprocesseur générés.
4. Exécuter le pipeline disponible :

```powershell
dvc status
dvc repro
```

`dvc repro` exécute Optuna pour les quatre modèles, suit les résultats des
essais et met à jour le modèle sélectionné. Aucun remote DVC n’est configuré;
les données préparées peuvent être régénérées localement depuis le dataset brut.

## Déploiement Docker Compose

Vérifier la configuration et l’état des services :

```powershell
docker compose config
docker compose ps
```

Démarrer les images déjà présentes :

```powershell
docker compose up -d
```

Au contrôle final du 25 novembre 2026, les images API et dashboard avaient été
reconstruites après l’entraînement Optuna et le correctif des métriques MLflow.
Les empreintes SHA-256 du modèle servi par l’API et du rapport consommé par le
dashboard correspondent aux artefacts locaux. Il n’est donc pas nécessaire de
reconstruire les images pour démarrer la version vérifiée.

Après une modification du code ou des artefacts, contrôler l’espace disque puis
reconstruire uniquement les services concernés. Pour une modification du
modèle ou des pages du dashboard :

```powershell
docker compose build api dashboard
docker compose up -d --no-deps api dashboard
```

Ces commandes ciblent uniquement l’API et le dashboard, sans supprimer les
volumes ni reconstruire Prometheus/Grafana. Au dernier contrôle, C: disposait
d’environ 3,3 Go libres et chaque image applicative occupait 1,16 Go. L’espace
nécessaire à un build dépend également des couches temporaires; vérifier la
place disponible avant toute reconstruction. Ne pas utiliser `down -v`,
`system prune` ni de commande de suppression de volume.

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
