"""Page d'accueil : architecture de la stack, état des services et modèle servi."""

import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import streamlit as st

try:
    from parts import config
except ImportError:  # streamlit run src/streamlit/parts/architecture.py
    import config

STACK_DOT = r"""
digraph stack {
  rankdir=LR;
  bgcolor="white";
  graph [fontname="Helvetica", fontsize=12, nodesep=0.35, ranksep=0.7, pad=0.2];
  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
         color="#5f6b7a", fontcolor="#1f2933", margin="0.15,0.07"];
  edge  [fontname="Helvetica", fontsize=9, color="#5f6b7a", fontcolor="#3e4c59"];

  user [label="Navigateur\nutilisateur", shape=oval, fillcolor="#fff4e5"];

  subgraph cluster_compose {
    label="docker compose"; labeljust="l"; fontsize=13;
    style="rounded,dashed"; color="#2b6cb0"; fontcolor="#2b6cb0";

    streamlit  [label="Streamlit\n:8501", fillcolor="#fde8e8"];
    api        [label="inference-api (FastAPI) :8000\n/predict · /train\n/train/simulation/{step}\n/metrics",
                fillcolor="#e3f2fd", color="#1565c0", penwidth=2];
    mlflow     [label="MLflow\n:5001 → 5000", fillcolor="#e8f5e9"];
    prometheus [label="Prometheus\n:9090", fillcolor="#fff3e0"];
    grafana    [label="Grafana\n:3000", fillcolor="#fff8e1"];
    airflow    [label="Airflow :8080\nsimulation_stream_dag", fillcolor="#ede7f6"];
    data       [label="data/\nCSV bruts, splits", shape=cylinder, fillcolor="#f2f4f7"];
    models     [label="models/\nmodel.joblib, history/", shape=cylinder, fillcolor="#f2f4f7"];
  }

  user       -> streamlit  [label="HTTP"];
  streamlit  -> api        [label="X-API-Key"];
  airflow    -> api        [label="POST /train/simulation/{step}"];
  prometheus -> api        [label="scrape /metrics"];
  grafana    -> prometheus [label="PromQL"];
  api        -> mlflow     [label="log des runs"];
  api        -> data       [dir=both];
  api        -> models     [dir=both];
}
"""  # noqa: E501

# health=None : pas de vérification (Streamlit sert cette page).
SERVICES = [
    {
        "name": "Streamlit",
        "health": None,
        "link": config.PUBLIC_STREAMLIT_URL,
        "role": "Présentation et démo (cette application)",
    },
    {
        "name": "inference-api",
        "health": f"{config.API_URL}/health",
        "link": f"{config.PUBLIC_API_URL}/docs",
        "role": "API FastAPI : prédiction, entraînements, /metrics",
    },
    {
        "name": "MLflow",
        "health": f"{config.MLFLOW_URL}/health",
        "link": config.PUBLIC_MLFLOW_URL,
        "role": "Suivi des runs (paramètres, métriques, artefacts)",
    },
    {
        "name": "Prometheus",
        "health": f"{config.PROMETHEUS_URL}/-/ready",
        "link": config.PUBLIC_PROMETHEUS_URL,
        "role": "Collecte des métriques de l'API (scrape de /metrics)",
    },
    {
        "name": "Grafana",
        "health": f"{config.GRAFANA_URL}/api/health",
        "link": config.PUBLIC_GRAFANA_URL,
        "role": "Dashboards ML et infra sur Prometheus",
    },
    {
        "name": "Airflow",
        "health": f"{config.AIRFLOW_URL}/health",
        "link": config.PUBLIC_AIRFLOW_URL,
        "role": "Orchestration : un step de simulation par run du DAG",
    },
]

CAMPAIGN_LABELS = {
    "idle": "inactive",
    "running": "en cours",
    "completed": "terminée",
    "failed": "échec",
}


def check_service(service: dict) -> dict:
    """Ligne du tableau d'état ; ne lève jamais d'exception."""
    row = {**service, "ok": False, "payload": None, "latency_ms": None}
    if service["health"] is None:
        return {**row, "ok": True}
    start = time.perf_counter()
    try:
        response = httpx.get(service["health"], timeout=config.HTTP_TIMEOUT)
    except httpx.TimeoutException:
        return {**row, "detail": f"timeout (> {config.HTTP_TIMEOUT:g} s)"}
    except Exception as exc:  # DNS, connexion refusée, etc.
        return {**row, "detail": f"{type(exc).__name__} : {exc}"[:80]}

    row["latency_ms"] = (time.perf_counter() - start) * 1000
    row["ok"] = response.status_code == 200
    row["detail"] = f"HTTP {response.status_code} · {row['latency_ms']:.0f} ms"
    try:
        row["payload"] = response.json()
    except ValueError:
        pass
    # Airflow répond 200 même si le scheduler est arrêté : on le vérifie aussi.
    if service["name"] == "Airflow" and row["ok"]:
        scheduler = (row["payload"] or {}).get("scheduler", {}).get("status")
        row["ok"] = scheduler == "healthy"
        row["detail"] += f" · scheduler : {scheduler}"
    return row


@st.cache_data(ttl=10, show_spinner="Vérification des services…")
def check_services() -> tuple[str, list[dict]]:
    with ThreadPoolExecutor(max_workers=len(SERVICES)) as pool:
        rows = list(pool.map(check_service, SERVICES))
    return time.strftime("%H:%M:%S"), rows


def show_architecture():
    st.title("🏗️ Architecture")
    st.caption(
        "Classification de produits Rakuten à partir du texte, servie par une API "
        "FastAPI ; entraînement, suivi, monitoring et orchestration dans une stack "
        "docker compose."
    )
    st.graphviz_chart(STACK_DOT, width="stretch")

    col_title, col_button = st.columns([6, 1], vertical_alignment="bottom")
    col_title.subheader("🩺 État des services")
    if col_button.button("🔄 Rafraîchir", width="stretch"):
        check_services.clear()
    checked_at, rows = check_services()
    with st.container(border=True):
        for row in rows:
            col_dot, col_name, col_role, col_badge, col_link = st.columns(
                [0.2, 1.2, 4.5, 0.7, 1], vertical_alignment="center"
            )
            col_dot.markdown("🟢" if row["ok"] else "🔴")
            col_name.markdown(f"**{row['name']}**")
            if row["ok"]:
                col_role.caption(row["role"])
                ms = row["latency_ms"]
                col_badge.badge("ici" if ms is None else f"{ms:.0f} ms", color="gray")
            else:
                col_role.caption(f"{row['role']} — {row['detail']}")
                col_badge.badge("KO", color="red")
            # "http://localhost:8000/docs" -> ":8000/docs"
            link_text = ":" + row["link"].rsplit(":", 1)[-1]
            col_link.markdown(f"[{link_text} ↗]({row['link']})")
    st.caption(f"Vérifié à {checked_at} depuis le conteneur Streamlit (cache 10 s).")

    st.subheader("🤖 Modèle servi")
    api = next(row for row in rows if row["name"] == "inference-api")
    if not api["ok"]:
        st.warning(f"inference-api injoignable : {api['detail']}")
    else:
        health = api["payload"]
        step = health.get("model_step")
        if not health.get("model_available"):
            step_label = "aucun"
            step_help = "Aucun models/model.joblib : lancer un entraînement."
        elif step is None:
            step_label = "complet"
            step_help = "Modèle entraîné sur toutes les données via POST /train."
        else:
            step_label = f"step {step}/{config.MAX_SIMULATION_STEP}"
            # cf. src/simulation/data_growth.py : step 0 = 50 % du flux, +5 % par step
            step_help = f"Modèle de simulation entraîné sur {50 + 5 * step} % du flux."

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Modèle disponible", "✅ oui" if health.get("model_available") else "❌ non"
        )
        col2.metric("Étape servie", step_label, help=step_help)
        col3.metric(
            "Entraînement en cours",
            "⏳ oui" if health.get("training_ongoing") else "non",
        )
        campaign = health.get("campaign_status")
        col4.metric(
            "Campagne de simulation",
            CAMPAIGN_LABELS.get(campaign, str(campaign)),
            help=(
                "Statut de la campagne steps 0 → 10 lancée par POST /train/simulation."
            ),
        )

    col_prep, col_train = st.columns(2)
    with col_prep.container(border=True):
        st.markdown(
            "**🧹 Preprocessing**  \n"
            "<small>`src/preprocessor.py`</small>\n"
            "- `designation` + `description` concaténées\n"
            "- HTML décodé, balises retirées (BeautifulSoup)\n"
            "- minuscules, accents supprimés (unidecode)\n"
            "- lemmatisation spaCy `fr_core_news_sm`\n"
            "- stopwords NLTK fr + en, tokens alphabétiques > 2 caractères",
            unsafe_allow_html=True,
        )
    with col_train.container(border=True):
        st.markdown(
            "**⚙️ Entraînement**  \n"
            "<small>`src/preprocessor.py`, `src/trainer.py`</small>\n"
            "- TF-IDF mots : lemmes, 1–2-grammes, min_df 5, max_df 0.8\n"
            "- TF-IDF caractères : `char_wb` 3–5, min_df 5, max_df 0.9\n"
            "- `sublinear_tf` sur les deux\n"
            "- `LinearSVC` C=1.0, class_weight balanced, random_state 42\n"
            "- validation 20 % stratifiée : accuracy, F1 macro, F1 pondéré",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_architecture()
