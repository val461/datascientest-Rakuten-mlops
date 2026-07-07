# datascientest-Rakuten-mlops

## Installation

Installer Docker.

Copier le fichier `.env.example` en un fichier `.env` à la racine du repo, et remplacer la valeur de `API_KEY` par une valeur de votre choix.

## Lancement

(Pas d'environnement virtuel à activer.)

`docker compose down; docker compose up --build`

## Endpoints

Voir http://localhost:8000/docs .

- GET /
- GET /docs → Swagger UI
- GET /health
- GET /data-status
- POST /train/simulation
- GET /train/simulation/status
- POST /train/simulation/{step}
- POST /predict → prédiction
- POST /train → réentraînement
- GET /metrics → pour prometheus

## Simulation de croissance des donnees

Le dataset est separe de maniere persistante en :

- 20 % de validation fixe ;
- 80 % de flux simulant l'arrivee des donnees.

Le premier modele utilise 50 % du flux. Chaque etape ajoute ensuite 5 % :

```text
step 0  -> 50 % du flux
step 1  -> 55 % du flux
...
step 10 -> 100 % du flux
```

Les index sont crees une seule fois dans `data/splits/`. Une empreinte SHA-256
empeche de reutiliser silencieusement le split si les CSV bruts changent.

Initialiser ou verifier le split sans entrainer :

```bash
docker compose run --rm inference-api python -m scripts.initialize_split
```

Entrainer une etape directement sans remplacer le modele servi :

```bash
docker compose run --rm inference-api python -m scripts.train_simulation_step 0
```

Ajouter `--deploy` uniquement pour promouvoir explicitement cette etape :

```bash
docker compose run --rm inference-api python -m scripts.train_simulation_step 0 --deploy
```

Lancer toute la campagne, avec deploiement du step 10 uniquement :

```bash
docker compose run --rm inference-api python -m scripts.run_simulation_campaign
```

Conserver l'ancien entrainement classique :

```bash
docker compose run --rm inference-api python -m scripts.train_full
```

Via l'API :

```text
POST /train
    Entrainement classique complet et deploiement.

POST /train/simulation
    Lance les steps 0 a 10 en arriere-plan.

GET /train/simulation/status
    Affiche l'avancement et les resultats disponibles.

POST /train/simulation/{step}
    Reproduit un step sans deploiement par defaut.

POST /train/simulation/{step}?deploy=true
    Reproduit et deploie explicitement un step.
```

L'endpoint authentifie `GET /data-status` expose le split et l'etape du
modele actuellement servi.

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

### Tests rapides dans Docker

Après la section [Lancement](#lancement), les tests unitaires et les tests d'intégration courts se lancent avec :

```bash
docker compose run --rm tests pytest \
  test_data_loader.py \
  test_trainer.py \
  test_api_contract.py \
  test_api.py \
  -v -k "not async_train"
```

Le test `test_async_train` est exclu de cette commande car il declenche un
veritable entrainement complet. Il doit etre lance volontairement lorsque le
temps d'execution et le remplacement du modele servi sont acceptes.

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


## Grafana

Grafana est accessible à la page http://localhost:3000 . Par défaut, le nom d'utilisateur et le mot de passe sont `admin`.

Ensuite, la [page suivante](http://localhost:3000/d/adp6w96/dashboard-1?orgId=1&from=now-30m&to=now&timezone=browser&refresh=10s)
montre le "dashboard 1".

Ce dashboard peut prendre environ 2 minutes avant de recevoir des données et d'afficher un graphique. Il est censé se rafraîchir automatiquement toutes les 10 secondes. Sinon, Grafana a un bouton "Refresh", à côté duquel est un bouton pour choisir l'intervalle de rafraîchissement.

La [page suivante](http://localhost:3000/a/grafana-metricsdrilldown-app/drilldown?var-ds=PBFA97CFB590B2093&uel_pid=grafana-metricsdrilldown-app&uel_epid=grafana%2Fdatasources%2Fconfig%2Fstatus&from=now-1h&to=now&timezone=browser&var-metrics_filters=&var-filters=&var-labelsWingman=%28none%29&layout=grid&filters-rule=&filters-prefix=&filters-suffix=&search_txt=&var-metrics-reducer-sort-by=default&filters-recent=&var-other_metric_filters=) montre beaucoup d'autres graphiques.

### Créer une visualisation

Ne pas oublier de passer en mode "Code" plutôt que "Builder" quand on tape une query en promQL.

### Partager un dashboard

Pour partager un dashboard via Grafana et git :
- exporter le dashboard en JSON (bouton "Export" dans Grafana)
- mettre le JSON dans `grafana/provisioning/dashboards`
- commit et push le JSON
- pull request.
