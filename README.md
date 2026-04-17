# datascientest-Rakuten-mlops

## Lancement

`docker compose down; docker compose up --build`

## Endpoints

http://localhost:8000/docs

- GET /docs → Swagger UI
- GET /health
- POST /predict → prédiction
- POST /train → réentraînement

## Tests with curl

### Health

```
curl -X 'GET' \
  'http://localhost:8000/health' \
  -H 'accept: application/json'
```

### Prédiction

```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "designation": "Folkmanis Puppets - Marionnette Et Theatre - Mini Turtle",
  "description": "Marionnette tortue miniature en tissu",
  "productid": 516376098,
  "imageid": 1019294171
}'
```

### (Ré-)entraînement

L'entraînement peut prendre 10 minutes.

```
curl -X 'POST' \
  'http://localhost:8000/train' \
  -H 'accept: application/json' \
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

Outside of Docker, in a terminal in the folder of this repository, run the following.
(The test may take 10 minutes because of training.)

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
├── Dockerfile             # pour container inference-api
└── docker-compose.yml
```

## MLflow

Chaque appel `POST /train` crée un run MLflow.

Puis ouvrir :

- `http://localhost:5001`

Le résultat de l'entraînement renvoie aussi :

- `mlflow_run_id`
- `tracking_uri`
- `experiment_name`
