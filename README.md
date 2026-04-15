# datascientest-Rakuten-mlops

## Lancement

`docker compose up --build`

Normalement, pas besoin de relancer la commande après édition des fichiers python.

## Endpoints

http://localhost:8000/docs

- GET /docs → Swagger UI
- GET /health
- POST /predict → prédiction
- POST /train → réentraînement

## Tests

### Health

```
curl -X 'GET' \
  'http://localhost:8000/health' \
  -H 'accept: application/json'
```

### (Ré-)entraînement

```
curl -X 'POST' \
  'http://localhost:8000/train' \
  -H 'accept: application/json' \
  -d ''
```

### Prédiction

```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 1,
  "sepal_width": 1,
  "petal_length": 1,
  "petal_width": 1
}'
```

## Arborescence

```
iris-api/
├── data/
│   ├── raw/               # mettre les CSV ici
│   │   ├── X_train.csv
│   │   └── Y_train.csv
│   └── preprocessed/      # artefacts éventuels de preprocessing
├── models/                # le modèle sauvegardé y sera créé
│   └── model.joblib
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # chargement dataset (CSV ou fallback Iris)
│   ├── preprocessor.py    # preprocessing
│   ├── trainer.py         # entraînement + sauvegarde du modèle
│   └── inference.py       # chargement + prédiction (utilisé par l'API)
├── main.py                # FastAPI
├── train.py               # script pour lancer l'entraînement manuellement si besoin
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
