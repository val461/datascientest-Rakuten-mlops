# datascientest-Rakuten-mlops

## Installation

Installer Docker.

Copier le fichier `.env.example` en un fichier `.env` à la racine du repo, et remplacer la valeur de `API_KEY` par une valeur de votre choix.

## Lancement

(Pas d'environnement virtuel à activer.)

`docker compose down; docker compose up --build`

## Endpoints

Voir http://localhost:8000/docs .

- GET /docs → Swagger UI
- GET /health
- POST /predict → prédiction
- POST /train → réentraînement
- GET /metrics → pour prometheus

## Tests with curl

(Les tests via pytest sont plus pratiques, voir section [Tests with pytest](#tests-with-pytest) plus bas.)

### Health

```
curl -X 'GET' \
  'http://localhost:8000/health' \
  -H 'accept: application/json'
```

### Prédiction

Remplacez `your_api_key` dans la commande par la clé définie dans votre `.env` (voir section [Installation](#installation) plus haut).

```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your_api_key' \
  -d '{
  "designation": "Folkmanis Puppets - Marionnette Et Theatre - Mini Turtle",
  "description": "Marionnette tortue miniature en tissu",
  "productid": 516376098,
  "imageid": 1019294171
}'
```

### (Ré-)entraînement

L'entraînement peut prendre 10 minutes.

Remplacez `your_api_key` dans la commande par la clé définie dans votre `.env` (voir section [Installation](#installation) plus haut).

```
curl -X 'POST' \
  'http://localhost:8000/train' \
  -H 'accept: application/json' \
  -H 'X-API-Key: your_api_key' \
  -d ''
```

Chaque entraînement journalise aussi :

- paramètres du modèle et du preprocessing
- métriques de validation
- artefacts du preprocessing
- modèle sauvegardé

dans un store MLflow local `mlruns/`.

## Tests with pytest

### First time: initialize the virtual environment for API testing

Outside of Docker, in a terminal in the folder of this repository, run the following.
(This **erases** the virtual environment `venv/` if it exists.)

```
rm -Rv venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Next times

Outside of Docker, in a terminal in the folder of this repository, run the following after the section [Lancement](#lancement).

(The tests may take 10 minutes because of training.)

```
source venv/bin/activate
pytest test_api.py -v
```

## Arborescence

```
datascientest-Rakuten-mlops/
├── data/
│   ├── raw/               # CSV source Rakuten
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   └── Y_train.csv
│   └── preprocessed/      # artefacts générés par le preprocessing TF-IDF
│       ├── vectorizer.joblib
│       ├── X_train_vectors.npz
│       ├── X_valid_vectors.npz
│       ├── y_train.csv
│       ├── y_valid.csv
│       ├── label_names.json
│       └── metadata.json
├── models/                # bundle classifieur + preprocessor sauvegardé
│   └── model.joblib
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # chargement des CSV Rakuten, split stratifié
│   ├── mlflow_tracking.py # configuration et logging MLflow
│   ├── preprocessor.py    # nettoyage texte, stopwords, lemmatisation, TF-IDF mot+caractère
│   ├── trainer.py         # entraînement LinearSVC, métriques
│   └── inference.py       # chargement + prédiction (utilisé par l'API)
├── main.py                # FastAPI pour les endpoints /predict, /train et /health
├── mlruns/                # store MLflow local (ignoré par git)
├── requirements.txt       # dépendances pour container inference-api
├── requirements-dev.txt   # dépendances pour test API hors de Docker
├── test_api.py            # test API via pytest hors de Docker
├── clean.sh               # script pour réinitialiser le repo et mlflow (effacer les artefacts)
├── .env.example           # template pour le fichier .env
├── .env                   # à créer pour définir la clé API
├── Dockerfile             # pour container inference-api
└── docker-compose.yml
```

## MLflow

Chaque appel `POST /train` crée un run MLflow.

Puis ouvrir :

- http://localhost:5001

Le résultat de l'entraînement renvoie aussi :

- `mlflow_run_id`
- `tracking_uri`
- `experiment_name`

## Prometheus

Prometheus charge les métriques de l'endpoint http://localhost:8000/metrics ,

et est accessible à la page http://localhost:9090/ .

Pour une explication des métriques exposées, ou pour ajouter des métriques, voir l'intro du readme du repo [prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator).
