# datascientest-Rakuten-mlops

## Lancement

`docker compose up --build`

## Endpoints

POST /predict → prédiction
POST /train → réentraînement
GET /health
GET /docs → Swagger UI

## Arborescence

iris-api/
├── data/
│   ├── raw/               # mettre le CSV ici
│   └── preprocessed/      # optionnel : artefacts de preprocessing
├── models/                # le modèle sauvegardé y sera créé
│   └── iris_model.joblib
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
