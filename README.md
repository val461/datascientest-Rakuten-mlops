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

```
curl -X 'POST' \
  'http://localhost:8000/train' \
  -H 'accept: application/json' \
  -d ''
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
│   ├── data_loader.py     # chargement des CSV Rakuten
│   ├── preprocessor.py    # nettoyage texte, stopwords, lemmatisation, TF-IDF mot+caractère
│   ├── trainer.py         # split stratifié, entraînement LinearSVC, métriques
│   └── inference.py       # chargement + prédiction (utilisé par l'API)
├── main.py                # FastAPI pour les endpoints /predict, /train et /health
├── train.py               # script pour lancer l'entraînement manuellement si besoin
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
