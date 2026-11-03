"""Page « Orchestration Airflow » : simulation de croissance des données.

Sources : README, airflow/dags/simulation_stream_dag.py et src/simulation/.
Seuls des appels GET sont faits vers l'API : aucun entraînement n'est lancé
depuis cette page.
"""

import re
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

from parts import config

# Mêmes valeurs que src/simulation/data_growth.py
INITIAL_AVAILABLE_RATIO = 0.50
RATIO_INCREMENT = 0.05
MAX_STEP = config.MAX_SIMULATION_STEP

# Mot de passe admin généré par `airflow standalone` au premier lancement, lu dans
# le volume airflow_home monté en lecture seule dans le conteneur Streamlit.
AIRFLOW_PASSWORD_FILE = Path("/airflow_home/standalone_admin_password.txt")
AIRFLOW_DAG_ID = "simulation_stream_dag"
AIRFLOW_STEP_VARIABLE = "simulation_current_step"

RUN_STATE_LABELS = {
    "success": "🟢 succès",
    "running": "🟡 en cours",
    "queued": "⚪ en attente",
    "failed": "🔴 échec",
}


def simulation_ratio(step: int) -> float:
    """Part du flux disponible à un step (même formule que data_growth.py)."""
    if step == MAX_STEP:
        return 1.0
    return round(INITIAL_AVAILABLE_RATIO + step * RATIO_INCREMENT, 2)


def get_airflow_interval_minutes() -> int:
    # La variable n'est définie que pour le service airflow : on la lit dans le
    # docker-compose.yml à la racine du repo (parts/ -> streamlit/ -> src/ -> racine).
    compose = Path(__file__).resolve().parents[3] / "docker-compose.yml"
    try:
        text = compose.read_text(encoding="utf-8")
        return int(re.search(r"AIRFLOW_INTERVAL_MINUTES:\s*[\"']?(\d+)", text).group(1))
    except (OSError, AttributeError):  # fichier absent ou ligne reformatée
        return 15


def read_airflow_password() -> str | None:
    try:
        return AIRFLOW_PASSWORD_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_stream_rows() -> int | None:
    """Taille du flux selon GET /data-status, ou None si l'API ne répond pas."""
    try:
        response = httpx.get(
            f"{config.API_URL}/data-status",
            headers=config.API_HEADERS,
            timeout=config.HTTP_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("stream_rows")
    except (httpx.HTTPError, ValueError):
        return None


# L'API REST d'Airflow n'accepte ici que l'authentification par session : on se
# connecte comme le ferait le navigateur (formulaire /login/ + cookie), avec le
# compte admin et le mot de passe lu dans le volume airflow_home.
@st.cache_data(ttl=15, show_spinner=False)
def fetch_airflow_status() -> dict:
    password = read_airflow_password()
    if not password:
        return {
            "ok": False,
            "error": "mot de passe admin introuvable "
            "(volume `airflow_home` non monté ?)",
        }
    try:
        with httpx.Client(
            base_url=config.AIRFLOW_URL,
            timeout=config.HTTP_TIMEOUT,
            follow_redirects=True,
        ) as client:
            login_page = client.get("/login/").text
            csrf = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', login_page)
            if not csrf:
                return {"ok": False, "error": "page de connexion Airflow inattendue"}
            client.post(
                "/login/",
                data={
                    "csrf_token": csrf.group(1),
                    "username": "admin",
                    "password": password,
                },
            )
            dag = client.get(f"/api/v1/dags/{AIRFLOW_DAG_ID}")
            if dag.status_code != 200:
                return {
                    "ok": False,
                    "error": f"HTTP {dag.status_code} sur "
                    f"`/api/v1/dags/{AIRFLOW_DAG_ID}` (connexion refusée ?)",
                }
            runs = client.get(
                f"/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns",
                params={"order_by": "-execution_date", "limit": 1},
            ).json()
            variable = client.get(f"/api/v1/variables/{AIRFLOW_STEP_VARIABLE}")
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": f"Airflow injoignable sur `{config.AIRFLOW_URL}` ({exc})",
        }

    return {
        "ok": True,
        "dag": dag.json(),
        "runs": runs,
        "step": variable.json()["value"] if variable.status_code == 200 else None,
        "fetched_at": datetime.now().strftime("%H:%M:%S"),
    }


def _local_time(iso_value: str | None, fmt: str = "%d/%m %H:%M") -> str:
    if not iso_value:
        return "—"
    return datetime.fromisoformat(iso_value).astimezone().strftime(fmt)


def show_airflow_status() -> None:
    status = fetch_airflow_status()
    if not status["ok"]:
        st.caption(f"Statut du DAG indisponible : {status['error']}")
        return

    dag, runs, step = status["dag"], status["runs"], status["step"]
    with st.container(border=True):
        col_dag, col_run, col_step, col_next = st.columns(4)

        if dag["has_import_errors"]:
            dag_label = "⚠️ erreur d'import"
            dag_help = "Le fichier du DAG ne se charge pas : voir l'UI Airflow."
        elif dag["is_paused"]:
            dag_label = "⏸️ en pause"
            dag_help = "Le DAG ne tourne pas tant qu'il n'est pas activé dans l'UI."
        else:
            dag_label = "▶️ actif"
            dag_help = "Le DAG est planifié et tourne selon son schedule."
        col_dag.metric("DAG", dag_label, help=dag_help)

        if runs.get("dag_runs"):
            run = runs["dag_runs"][0]
            started = _local_time(run["start_date"], "%d/%m/%Y %H:%M:%S")
            col_run.metric(
                "Dernier run",
                RUN_STATE_LABELS.get(run["state"], run["state"]),
                help=f"{run['dag_run_id']} · {run['run_type']} · démarré le "
                f"{started} · {runs['total_entries']} run(s) au total",
            )
        else:
            col_run.metric("Dernier run", "—", help="Aucun run pour l'instant.")

        col_step.metric(
            "Step courant",
            "—" if step is None else f"{step} / {MAX_STEP}",
            help=f"Variable Airflow `{AIRFLOW_STEP_VARIABLE}` : prochain step que "
            "le DAG entraînera."
            + (" Elle est créée à la fin du premier run." if step is None else ""),
        )

        col_next.metric(
            "Prochain run",
            "—" if dag["is_paused"] else _local_time(dag["next_dagrun"]),
            help="Heure planifiée du prochain run (heure locale) ; "
            "« — » tant que le DAG est en pause.",
        )
        st.caption(
            f"Lu via l'API REST d'Airflow à {status['fetched_at']} (cache 15 s)."
        )


def show_principle(interval_minutes: int) -> None:
    stream_rows = fetch_stream_rows()

    st.header("Principe de la simulation")
    st.caption(
        "Plutôt que d'entraîner le modèle une seule fois sur 100 % des données, "
        "on reproduit l'arrivée progressive de nouvelles données dans le temps."
    )

    col_split, col_steps, col_dag = st.columns(3)
    with col_split.container(border=True, height="stretch"):
        st.markdown(
            "**✂️ Split persistant**\n"
            "- validation **stratifiée 20 %**, identique à tous les steps\n"
            "- flux **80 % mélangé** : l'ordre simule l'arrivée des données\n"
            "- index + empreinte SHA-256 des CSV dans `data/splits/`, "
            "revalidés à chaque entraînement"
        )
    with col_steps.container(border=True, height="stretch"):
        st.markdown(
            f"**📶 {MAX_STEP + 1} steps cumulatifs**\n"
            f"- step 0 = 50 % du flux, +5 % par step, step {MAX_STEP} = 100 %\n"
            "- même bloc de validation à chaque step : accuracy, F1 macro, F1 pondéré\n"
            "- par step : un run MLflow `LinearSVC-step-XX-ratio-R.RR` "
            "et un `models/history/model_step_XX.joblib`"
        )
    with col_dag.container(border=True, height="stretch"):
        st.markdown(
            "**⏱️ Un step par run du DAG**\n"
            f"- toutes les **{interval_minutes} min** (`AIRFLOW_INTERVAL_MINUTES`)\n"
            "- step courant persisté dans la Variable Airflow "
            "`simulation_current_step` : le DAG reprend là où il s'était arrêté\n"
            f"- modèle servi par `/predict` remplacé au **step {MAX_STEP}** seulement"
        )

    # Frise des steps : part du flux et lignes d'entraînement
    for step, col in enumerate(st.columns(MAX_STEP + 1)):
        ratio = simulation_ratio(step)
        with col.container(border=True, height="stretch"):
            st.caption(f"step {step}")
            pct = f"**{ratio * 100:.0f} %**"
            st.markdown(f":blue[{pct}]" if step == MAX_STEP else pct)
            if stream_rows:
                st.caption(f"{int(stream_rows * ratio):,}".replace(",", " "))
            if step == MAX_STEP:
                st.caption(":blue[servi]")


def build_dag_dot(interval_minutes: int) -> str:
    """Graphe du DAG simulation_stream_dag (noms de tâches identiques au code)."""
    return f"""
digraph simulation_stream_dag {{
    rankdir=LR;
    bgcolor="transparent";
    nodesep=0.35;
    ranksep=0.45;
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
          fillcolor="#dbeafe", color="#2563eb", fontcolor="#0f172a", margin="0.15,0.08"];
    edge [color="#64748b", fontcolor="#64748b", fontname="Helvetica", fontsize=10];

    schedule [label="schedule\\ntoutes les {interval_minutes} min\\ncatchup=False\\nmax_active_runs=1",
              shape=note, fillcolor="#fef3c7", color="#d97706"];
    check_health [label="check_health\\nGET /health"];
    get_current_step [label="get_current_step\\nlit la Variable\\nsimulation_current_step"];
    advance_simulation [label="advance_simulation\\nPOST /train/simulation/{{step}}"];
    save_next_step [label="save_next_step\\nécrit step + 1\\ndans la Variable"];
    failed [label="run en échec\\nstep inchangé", fillcolor="#fee2e2", color="#dc2626"];
    skipped [label="skip\\nAirflowSkipException", fillcolor="#e5e7eb", color="#6b7280"];

    schedule -> check_health;
    check_health -> get_current_step [label="API OK"];
    check_health -> failed [label="API KO", style=dashed];
    get_current_step -> advance_simulation [label="step"];
    advance_simulation -> save_next_step [label="step + 1"];
    advance_simulation -> skipped [label="step > {MAX_STEP}", style=dashed];
}}
"""  # noqa: E501


def show_dag(interval_minutes: int) -> None:
    col_title, col_refresh = st.columns([6, 1], vertical_alignment="bottom")
    col_title.header("Le DAG `simulation_stream_dag`")
    if col_refresh.button("🔄 Rafraîchir", key="airflow_refresh", width="stretch"):
        fetch_airflow_status.clear()
    show_airflow_status()
    st.graphviz_chart(build_dag_dot(interval_minutes), width="stretch")

    cards = [
        (
            "🕒 Planification",
            f"toutes les **{interval_minutes} min** (`AIRFLOW_INTERVAL_MINUTES`) · "
            "`catchup=False` · `max_active_runs=1` : jamais deux steps en parallèle",
        ),
        (
            "🩺 Garde-fous",
            "`check_health` doit réussir avant de lire le step, sinon le run échoue "
            "et le step n'avance pas · `AirflowSkipException` si le step dépasse "
            f"{MAX_STEP}",
        ),
        (
            "🔌 Connexion",
            "connexion HTTP `training_api` (`AIRFLOW_CONN_TRAINING_API` → "
            "`inference-api:8000`) · clé envoyée dans l'en-tête `X-API-Key`",
        ),
        (
            "⏸️ Créé en pause",
            "`DAGS_ARE_PAUSED_AT_CREATION` : l'activer dans l'UI, puis laisser "
            "tourner ou *Trigger DAG* · step courant visible dans *Admin › Variables*",
        ),
    ]
    for col, (title, body) in zip(st.columns(4), cards, strict=True):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.markdown(body)

    col_link, col_password, col_note = st.columns(
        [1, 1.2, 3], vertical_alignment="center"
    )
    col_link.link_button(
        "Ouvrir l'UI Airflow", config.PUBLIC_AIRFLOW_URL, icon="🌐", width="stretch"
    )
    password = read_airflow_password()
    if password:
        with col_password.popover("🔑 Mot de passe admin", width="stretch"):
            st.code(password, language=None)
        col_note.markdown(
            "Compte `admin` · mot de passe généré par `airflow standalone` au premier "
            "lancement, lu dans le volume `airflow_home`."
        )
    else:
        col_password.markdown("Compte `admin`")
        col_note.markdown(
            "mot de passe généré au premier lancement : "
            "`docker logs rakuten-airflow | grep password`"
        )


def show_orchestration():
    st.title("⏱️ Orchestration Airflow")
    interval_minutes = get_airflow_interval_minutes()
    show_principle(interval_minutes)
    st.divider()
    show_dag(interval_minutes)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_orchestration()
