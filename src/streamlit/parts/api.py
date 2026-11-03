# parts/api.py
"""Page « API FastAPI » : endpoints, état en direct et prédiction libre."""

import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

# Permet aussi `streamlit run src/streamlit/parts/api.py` (hors app.py) :
# `parts.*` a besoin de src/streamlit/ et `parts.demo` de la racine du repo.
_STREAMLIT_DIR = Path(__file__).resolve().parent.parent
for _path in (_STREAMLIT_DIR, _STREAMLIT_DIR.parent.parent):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from parts.config import (  # noqa: E402
    API_HEADERS,
    API_URL,
    HTTP_TIMEOUT,
    MAX_SIMULATION_STEP,
    PUBLIC_API_URL,
)

try:
    from parts.demo import get_category_mapping
except Exception:  # dépendances de la démo absentes : la page reste utilisable
    def get_category_mapping():
        return {}


# Le premier /predict peut être lent (chargement du modèle) : timeout large.
PREDICT_TIMEOUT = 60.0

# Exemple du README (section « Tests with curl »)
EXAMPLE_DESIGNATION = "Folkmanis Puppets - Marionnette Et Theatre - Mini Turtle"
EXAMPLE_DESCRIPTION = "Marionnette tortue miniature en tissu"

# Endpoints de main.py dans l'ordre d'affichage, avec leur description (FR) ;
# sert aussi de liste de repli si /openapi.json est injoignable. Un endpoint
# inconnu s'affiche en dernier, avec le résumé de son schéma OpenAPI.
ENDPOINT_DESCRIPTIONS = {
    ("POST", "/predict"): "Prédit le code produit (prdtypecode) à partir du texte",
    ("POST", "/train"):
        "Entraîne le modèle complet sur toutes les données et le déploie",
    ("POST", "/train/simulation"):
        "Lance en tâche de fond la campagne de simulation (steps 0 → 10)",
    ("POST", "/train/simulation/{step}"):
        "Entraîne une seule étape de simulation (déploiement optionnel)",
    ("GET", "/train/simulation/status"): "Progression de la campagne de simulation",
    ("GET", "/health"):
        "État du service : modèle chargé, entraînement en cours, campagne",
    ("GET", "/data-status"):
        "Métadonnées du split (validation fixe / flux) et du modèle servi",
    ("GET", "/metrics"): "Métriques Prometheus (requêtes, latences…)",
}
# Endpoints sans X-API-Key, pour la liste de repli (en direct : lu dans le schéma)
PUBLIC_PATHS = {"/health", "/metrics"}

# statut de campagne -> (libellé, couleur de badge)
CAMPAIGN_BADGES = {
    "idle": ("inactive", "gray"),
    "running": ("en cours", "orange"),
    "completed": ("terminée", "green"),
    "failed": ("échec", "red"),
}


# --------------------------------------------------------------------------
# Appels HTTP
# --------------------------------------------------------------------------
def _get_json(path: str, auth: bool = False) -> dict:
    """GET sur l'API sans jamais lever : {ok, status_code, data, error}."""
    url = f"{API_URL}{path}"
    try:
        response = httpx.get(
            url, headers=API_HEADERS if auth else None, timeout=HTTP_TIMEOUT
        )
    except httpx.TimeoutException:
        return {"ok": False, "status_code": None, "data": None,
                "error": f"délai dépassé ({HTTP_TIMEOUT:.0f} s) en contactant {url}"}
    except httpx.HTTPError as exc:
        return {"ok": False, "status_code": None, "data": None,
                "error": f"impossible de joindre {url} ({exc})"}

    try:
        data = response.json()
    except ValueError:
        data = response.text
    error = None
    if not response.is_success:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        error = f"HTTP {response.status_code} : {detail}"
    return {"ok": response.is_success, "status_code": response.status_code,
            "data": data, "error": error}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_endpoints() -> tuple[list[tuple[str, str, bool, str]], str | None]:
    """(méthode, chemin, protégé, description) lus en direct sur /openapi.json,
    ou liste de repli et message d'erreur si le schéma est injoignable."""
    result = _get_json("/openapi.json")
    if not result["ok"]:
        fallback = [
            (method, path, path not in PUBLIC_PATHS, description)
            for (method, path), description in ENDPOINT_DESCRIPTIONS.items()
        ]
        return fallback, result["error"]

    endpoints = []
    for path, operations in result["data"]["paths"].items():
        for method, operation in operations.items():
            key = (method.upper(), path)
            description = ENDPOINT_DESCRIPTIONS.get(key, operation["summary"])
            endpoints.append((*key, "security" in operation, description))
    return endpoints, None


@st.cache_data(ttl=5, show_spinner=False)
def fetch_health() -> dict:
    return _get_json("/health")


@st.cache_data(ttl=5, show_spinner=False)
def fetch_data_status() -> dict:
    return _get_json("/data-status", auth=True)


@st.cache_data(ttl=5, show_spinner=False)
def fetch_campaign_status() -> dict:
    return _get_json("/train/simulation/status", auth=True)


def call_predict(designation: str, description: str | None) -> dict:
    """POST /predict. Lève httpx.HTTPError en cas d'erreur réseau."""
    payload = {"designation": designation, "description": description}
    start = time.perf_counter()
    response = httpx.post(
        f"{API_URL}/predict", headers=API_HEADERS, json=payload,
        timeout=PREDICT_TIMEOUT,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text}
    return {"status_code": response.status_code, "data": data,
            "elapsed_ms": elapsed_ms, "payload": payload}


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------
def _int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _format_detail(detail) -> str:
    """`detail` FastAPI : chaîne, ou liste d'erreurs de validation (422)."""
    if isinstance(detail, list):
        return " ; ".join(
            f"{'.'.join(map(str, err['loc']))} : {err['msg']}" for err in detail
        )
    return str(detail)


def _clear_live_cache():
    fetch_health.clear()
    fetch_data_status.clear()
    fetch_campaign_status.clear()


def group_endpoints(endpoints: list[tuple]) -> dict[str, list[tuple]]:
    """Regroupe les endpoints par thème. Ils sont triés dans l'ordre de
    ENDPOINT_DESCRIPTIONS (inconnus à la fin), ce qui ordonne aussi les groupes."""
    rank = {key: i for i, key in enumerate(ENDPOINT_DESCRIPTIONS)}
    groups: dict[str, list[tuple]] = {}
    for endpoint in sorted(endpoints, key=lambda e: rank.get(e[:2], len(rank))):
        path = endpoint[1]
        if path.startswith("/predict"):
            group = "Prédiction"
        elif path.startswith("/train"):
            group = "Entraînement"
        else:
            group = "État et métriques"
        groups.setdefault(group, []).append(endpoint)
    return groups


def show_endpoints():
    col_title, col_swagger, col_redoc = st.columns(
        [5, 1, 1], vertical_alignment="bottom"
    )
    col_title.subheader("📋 Endpoints")
    col_swagger.link_button("📖 Swagger UI", f"{PUBLIC_API_URL}/docs", width="stretch")
    col_redoc.link_button("📘 ReDoc", f"{PUBLIC_API_URL}/redoc", width="stretch")

    endpoints, error = fetch_endpoints()
    if error:
        st.warning(
            f"Schéma OpenAPI indisponible ({error}). "
            "Liste statique dérivée de `main.py`."
        )

    with st.container(border=True):
        for group, rows in group_endpoints(endpoints).items():
            st.markdown(f":gray[**{group}**]")
            for method, path, auth, description in rows:
                col_method, col_path, col_desc, col_lock = st.columns(
                    [0.7, 2.4, 5, 0.4], vertical_alignment="center"
                )
                col_method.badge(method, color="green" if method == "GET" else "violet")
                col_path.markdown(f"`{path}`")
                col_desc.caption(description)
                if auth:
                    col_lock.markdown("🔒")

    st.markdown(
        "🔑 En-tête :gray-badge[X-API-Key] (clé définie dans `.env`) · "
        "sans clé :orange-badge[401] · clé invalide :red-badge[403] · "
        "entraînement déjà en cours :orange-badge[409]"
    )


def _badge(text, color: str = "gray") -> str:
    return f":{color}-badge[{text}]"


def _kv(label: str, value_md: str) -> None:
    col_key, col_value = st.columns([1, 1.7], vertical_alignment="center")
    col_key.caption(label)
    col_value.markdown(value_md)


def split_bar_html(
    validation_rows: int, stream_rows: int, available_ratio: float
) -> str:
    """Barre empilée : validation fixe | flux disponible | flux à venir."""
    total = validation_rows + stream_rows
    validation_pct = validation_rows / total * 100
    available_pct = stream_rows * available_ratio / total * 100
    future_pct = max(0.0, 100 - validation_pct - available_pct)
    return (
        '<div style="display:flex;height:14px;border-radius:6px;overflow:hidden;'
        'margin:6px 0 4px">'
        f'<div style="width:{validation_pct:.1f}%;background:#7F77DD"></div>'
        f'<div style="width:{available_pct:.1f}%;background:#1D9E75"></div>'
        f'<div style="width:{future_pct:.1f}%;background:#1D9E75;opacity:.3"></div>'
        "</div>"
        '<div style="display:flex;justify-content:space-between;'
        'font-size:12px;opacity:.75">'
        f"<span>🟪 validation fixe · {_int(validation_rows)} "
        f"({validation_pct:.0f} %)</span>"
        f"<span>🟩 flux · {_int(stream_rows)} ({100 - validation_pct:.0f} %) · "
        f"{available_ratio:.0%} disponible</span>"
        "</div>"
    )


def show_service_card(h: dict, api_key_ok: bool | None) -> None:
    st.markdown("**Service** · `GET /health`")

    step = h.get("model_step")
    if not h.get("model_available"):
        model_md = _badge("absent", "red") + " aucun `models/model.joblib`"
    elif step is None:
        model_md = _badge("chargé", "green") + " modèle complet (`POST /train`)"
    else:
        model_md = _badge("chargé", "green") + f" step {step} / {MAX_SIMULATION_STEP}"
    _kv("Modèle", model_md)

    if h.get("training_ongoing"):
        _kv("Entraînement", _badge("en cours", "orange") + " un nouvel appel → `409`")
    else:
        _kv("Entraînement", _badge("aucun en cours"))

    campaign = h["campaign_status"]
    _kv("Campagne", _badge(*CAMPAIGN_BADGES.get(campaign, (campaign or "—", "gray"))))

    if api_key_ok is None:
        _kv("Clé API", _badge("non vérifiée"))
    elif api_key_ok:
        _kv("Clé API", _badge("valide", "green"))
    else:
        _kv("Clé API", _badge("refusée", "red") + " vérifier `API_KEY` dans `.env`")

    if campaign == "running":
        status = fetch_campaign_status()
        if status["ok"]:
            c = status["data"]
            done, total = c["completed_steps"], c["total_steps"]
            st.progress(
                done / total,
                text=f"{done}/{total} étapes terminées · "
                f"étape en cours : {c['current_step']}",
            )


def show_data_card(data_status: dict) -> None:
    st.markdown("**Données** · `GET /data-status` 🔒")
    if not data_status["ok"]:
        if data_status["status_code"] in (401, 403):
            st.warning(f"Accès refusé ({data_status['error']}).")
        else:
            st.warning(f"Indisponible : {data_status['error']}")
        return
    d = data_status["data"]
    if not d["split_initialized"]:
        st.info(
            "Split non initialisé : aucun entraînement n'a encore créé "
            "le split validation / flux."
        )
        return

    ratio = d.get("current_available_ratio", 1.0)  # absent : modèle complet servi
    st.markdown(
        split_bar_html(d["validation_rows"], d["stream_rows"], ratio),
        unsafe_allow_html=True,
    )
    _kv("Lignes totales", _int(d["total_rows"]))
    step = d.get("current_model_step")
    if step is None:
        _kv("Modèle servi", "modèle complet (`POST /train`), hors simulation")
    else:
        _kv(
            "Modèle servi",
            f"step {step} → {_int(d['current_training_rows'])} lignes d'entraînement",
        )


def show_live_state():
    col_title, col_button = st.columns([6, 1], vertical_alignment="bottom")
    col_title.subheader("📡 État en direct")
    col_button.button(
        "🔄 Rafraîchir", on_click=_clear_live_cache, key="api_refresh", width="stretch"
    )

    health = fetch_health()
    if not health["ok"]:
        st.error(f"API injoignable : {health['error']}")
        return

    data_status = fetch_data_status()
    if data_status["ok"]:
        api_key_ok = True
    elif data_status["status_code"] in (401, 403):
        api_key_ok = False
    else:
        api_key_ok = None

    col_service, col_data = st.columns([1, 1.2], gap="medium")
    with col_service.container(border=True, height="stretch"):
        show_service_card(health["data"], api_key_ok)
    with col_data.container(border=True, height="stretch"):
        show_data_card(data_status)

    st.caption(f"Mis à jour à {datetime.now():%H:%M:%S} (cache 5 s).")


def show_prediction_form():
    st.subheader("🔮 Prédiction en texte libre")
    st.markdown(
        "Saisissez un produit : le texte est envoyé à **`POST /predict`** "
        "(TF-IDF mots + caractères, puis LinearSVC)."
    )

    with st.form("api_predict_form"):
        designation = st.text_input("Désignation *", value=EXAMPLE_DESIGNATION)
        description = st.text_area(
            "Description (optionnelle)", value=EXAMPLE_DESCRIPTION
        )
        submitted = st.form_submit_button("Prédire", type="primary")

    if submitted:
        st.session_state.pop("api_last_prediction", None)
        if not designation.strip():
            st.warning("La désignation est obligatoire.")
        else:
            try:
                st.session_state["api_last_prediction"] = call_predict(
                    designation.strip(), description.strip() or None
                )
            except httpx.TimeoutException:
                st.error(
                    f"Délai dépassé ({PREDICT_TIMEOUT:.0f} s) en contactant l'API "
                    f"sur `{API_URL}/predict`."
                )
            except httpx.HTTPError as exc:
                st.error(f"Impossible de joindre l'API sur `{API_URL}` : {exc}")

    if result := st.session_state.get("api_last_prediction"):
        show_prediction_result(result)


def show_prediction_result(result: dict):
    status_code, data = result["status_code"], result["data"]
    detail = _format_detail(data.get("detail"))

    if status_code == 200:
        code = data["prediction"]
        label = get_category_mapping().get(code, "libellé inconnu")
        col_code, col_time = st.columns(2)
        col_code.metric("Code prédit (prdtypecode)", code, border=True)
        col_time.metric(
            "Temps de réponse", f"{result['elapsed_ms']:.0f} ms", border=True,
            help="Aller-retour HTTP complet mesuré depuis le conteneur Streamlit.",
        )
        st.success(f"**{code}** — {label}")
    elif status_code == 503:
        st.error(f"Aucun modèle disponible (503) : {detail}")
        st.info(
            "Entraînez d'abord un modèle avec `POST /train` "
            f"(via [Swagger]({PUBLIC_API_URL}/docs) ou la commande curl du README)."
        )
    elif status_code == 403:
        st.error(
            "Clé API invalide (403) : vérifiez `API_KEY` dans `.env`. "
            f"Détail : {detail}"
        )
    elif status_code == 401:
        st.error(
            "Clé API manquante (401) : `API_KEY` n'est pas définie pour le "
            f"conteneur Streamlit. Détail : {detail}"
        )
    else:
        st.error(f"Erreur HTTP {status_code} : {detail}")

    with st.expander("Requête / réponse brutes"):
        st.markdown(f"`POST {API_URL}/predict` → **{status_code}** "
                    f"en {result['elapsed_ms']:.0f} ms")
        st.json({"requête": result["payload"], "réponse": data})


def show_api():
    st.title("🔌 API FastAPI")
    st.markdown(
        "Le modèle est servi par **`inference-api`**, une API FastAPI qui expose "
        "la prédiction, les entraînements (complet et simulation de croissance "
        "des données), l'état du service et des métriques Prometheus."
    )

    show_endpoints()
    st.divider()
    show_live_state()
    st.divider()
    show_prediction_form()


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_api()
