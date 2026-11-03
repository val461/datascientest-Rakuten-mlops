"""URLs et constantes partagées par les pages Streamlit.

Deux familles d'URLs :
- les URLs *internes* servent aux appels HTTP faits depuis le conteneur
  Streamlit (noms de services du réseau docker compose) ;
- les URLs *publiques* servent aux liens cliquables affichés à l'utilisateur
  (ports exposés sur localhost par docker compose).
"""

import os

API_KEY = os.getenv("API_KEY")
API_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# --- URLs internes (réseau docker compose) ---------------------------------
API_URL = os.getenv("API_BASE_URL", "http://inference-api:8000")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow:8080")

# --- URLs publiques (navigateur de l'utilisateur) --------------------------
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000")
PUBLIC_MLFLOW_URL = os.getenv("PUBLIC_MLFLOW_URL", "http://localhost:5001")
PUBLIC_PROMETHEUS_URL = os.getenv("PUBLIC_PROMETHEUS_URL", "http://localhost:9090")
PUBLIC_GRAFANA_URL = os.getenv("PUBLIC_GRAFANA_URL", "http://localhost:3000")
PUBLIC_AIRFLOW_URL = os.getenv("PUBLIC_AIRFLOW_URL", "http://localhost:8080")
PUBLIC_STREAMLIT_URL = os.getenv("PUBLIC_STREAMLIT_URL", "http://localhost:8501")

# --- Grafana ---------------------------------------------------------------
# uid -> titre, tels que provisionnés dans grafana/provisioning/dashboards/
GRAFANA_DASHBOARDS = {
    "adp6w96": "dashboard ML",
    "adwgslf": "dashboard infra",
}

# --- MLflow ----------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "rakuten-text-classification"

# --- Simulation de croissance des données ---------------------------------
MAX_SIMULATION_STEP = 10  # step 0 = 50 % du flux, step 10 = 100 %

# Timeout par défaut des appels HTTP depuis les pages (secondes)
HTTP_TIMEOUT = 5.0
