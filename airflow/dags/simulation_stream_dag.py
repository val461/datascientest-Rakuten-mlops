"""
DAG Airflow : simulation de l'arrivée progressive des données d'entraînement.

Logique :
1. On vérifie que l'API est en bonne santé via GET /health.
2. On récupère le "step" courant de la simulation (persisté dans une
   Airflow Variable, car il doit survivre d'une exécution du DAG à l'autre).
3. On appelle POST /train/simulation/{step} pour faire "avancer" la
   simulation (comme si suffisamment de temps s'était écoulé pour que de
   nouvelles données du flux deviennent disponibles).
4. On incrémente le step pour la prochaine exécution.
5. Une fois le step maximum atteint (10 = 100% des données disponibles),
   le DAG s'arrête de lui-même (skip) au lieu de rappeler l'API inutilement.

Hypothèses à adapter à votre infra :
- Une Airflow Connection HTTP nommée "training_api" pointe vers l'API
  (host + éventuels headers d'auth). A créer dans Admin > Connections.
- Le step courant est stocké dans la Variable Airflow
  "simulation_current_step" (créée automatiquement au premier run si absente).
- Le DAG tourne une fois par jour (@daily) : à changer selon la fréquence
  réelle souhaitée pour "simuler que du temps a passé".
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from airflow.providers.http.hooks.http import HttpHook

MAX_SIMULATION_STEP = 10
VARIABLE_NAME = "simulation_current_step"
HTTP_CONN_ID = "training_api"

logger = logging.getLogger(__name__)


@dag(
    dag_id="simulation_stream_dag",
    description="Avance pas à pas la simulation de flux de données d'entraînement",
    schedule=timedelta(minutes=5),  # à ajuster : @yearly, @daily, @hourly, timedelta(minutes=5), cron custom, etc.
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,  # évite deux avancées de step en parallèle
    tags=["training", "simulation"],
    is_paused_upon_creation=True,
)
def simulation_stream_dag():
    @task
    def check_health() -> None:
        """Vérifie que l'API répond correctement sur /health."""
        hook = HttpHook(method="GET", http_conn_id=HTTP_CONN_ID)
        response = hook.run(endpoint="health")
        response.raise_for_status()
        logger.info("Health check OK : %s", response.text)

    @task
    def get_current_step() -> int:
        """Récupère le step courant de la simulation, persisté entre runs."""
        step = int(Variable.get(VARIABLE_NAME, default_var=0))
        if step > MAX_SIMULATION_STEP:
            step = MAX_SIMULATION_STEP
        logger.info("Step courant de la simulation : %s", step)
        return step

    @task
    def advance_simulation(step: int) -> int:
        """Appelle /train/simulation/{step} puis renvoie le prochain step."""
        if step > MAX_SIMULATION_STEP:
            raise AirflowSkipException(
                f"Step {step} au-delà du maximum ({MAX_SIMULATION_STEP}), "
                "simulation déjà terminée."
            )

        hook = HttpHook(method="POST", http_conn_id=HTTP_CONN_ID)
        response = hook.run(endpoint=f"train/simulation/{step}")
        response.raise_for_status()
        logger.info(
            "Simulation avancée au step %s. Réponse API : %s",
            step,
            response.text,
        )
        return step + 1

    @task
    def save_next_step(next_step: int) -> None:
        """Persiste le prochain step pour la prochaine exécution du DAG."""
        if next_step > MAX_SIMULATION_STEP:
            next_step = MAX_SIMULATION_STEP
            logger.info(
                "Step maximum atteint (%s). Le DAG ne progressera plus.",
                MAX_SIMULATION_STEP,
            )
        Variable.set(VARIABLE_NAME, next_step)
        logger.info("Prochain step enregistré : %s", next_step)

    current_step = get_current_step()
    next_step = advance_simulation(current_step)

    check_health() >> current_step
    save_next_step(next_step)


simulation_stream_dag()
