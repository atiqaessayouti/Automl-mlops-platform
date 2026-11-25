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

Le dashboard français a été reconstruit le 23 novembre 2026. Après le dernier
entraînement Optuna, les conteneurs API et dashboard n’ont pas été reconstruits
et ne servent donc pas les artefacts locaux les plus récents. Lorsque l’espace
disque a été revérifié et qu’un build est explicitement autorisé, reconstruire
uniquement ces deux services :

```powershell
docker compose build api dashboard
docker compose up -d --no-deps api dashboard
```

Ces commandes ciblent les services nécessaires à la mise à jour du modèle et
de ses métriques. Elles ne suppriment pas les volumes ; ne pas utiliser
`down -v`, `system prune` ou une commande de suppression de volume. Le dernier
contrôle indiquait environ 4,99 Go libres sur C: et des images API/dashboard
d’environ 1,16 Go chacune; cela ne garantit pas à lui seul la capacité d’un
build, qui dépend aussi des couches intermédiaires.

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
