"""Page « Monitoring » : métriques Prometheus de l'API et dashboards Grafana.

Les graphiques interrogent l'API HTTP de Prometheus depuis le conteneur Streamlit
(URL interne ``PROMETHEUS_URL``) ; les dashboards Grafana sont intégrés en iframe
avec l'URL publique ``PUBLIC_GRAFANA_URL``, car c'est le navigateur qui les charge.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import altair as alt
import httpx
import numpy as np
import pandas as pd
import streamlit as st
import yaml

try:
    from parts.config import (
        GRAFANA_DASHBOARDS,
        HTTP_TIMEOUT,
        PROMETHEUS_URL,
        PUBLIC_GRAFANA_URL,
        PUBLIC_PROMETHEUS_URL,
    )
except ImportError:  # exécution directe du fichier (bloc __main__)
    from config import (
        GRAFANA_DASHBOARDS,
        HTTP_TIMEOUT,
        PROMETHEUS_URL,
        PUBLIC_GRAFANA_URL,
        PUBLIC_PROMETHEUS_URL,
    )


CACHE_TTL = 15  # secondes, = intervalle de scrape de Prometheus
REPO_ROOT = Path(__file__).resolve().parents[3]  # /app dans Docker
PROMETHEUS_CONFIG_FILE = REPO_ROOT / "prometheus.yml"

# Fenêtre affichée -> (durée en s, pas des points en s, plage Grafana)
WINDOWS = {
    "15 min": (15 * 60, 15, "15m"),
    "30 min": (30 * 60, 15, "30m"),
    "1 h": (60 * 60, 30, "1h"),
    "3 h": (3 * 60 * 60, 60, "3h"),
}
DEFAULT_WINDOW = "30 min"

# Palette catégorielle, ordre fixe : une couleur suit une entité (un endpoint,
# un quantile...), jamais son rang, pour rester stable d'un rafraîchissement à l'autre.
PALETTE = [
    "#2a78d6",  # bleu
    "#eb6834",  # orange
    "#1baf7a",  # turquoise
    "#eda100",  # jaune
    "#e87ba4",  # magenta
    "#008300",  # vert
    "#6250d6",  # violet
    "#e34948",  # rouge
]
NEUTRAL = "#8a8986"  # gris lisible sur fond clair comme sombre

TOTAL = "Total"
OTHER = "autres"
SIMULATION = "/train/simulation/*"
# Endpoints suivis un par un ; les autres (/, /docs, /openapi.json...) sont regroupés.
HANDLER_COLORS = {
    "/predict": PALETTE[0],
    "/health": PALETTE[1],
    "/metrics": PALETTE[2],
    "/train": PALETTE[3],
    SIMULATION: PALETTE[4],
    "/data-status": PALETTE[5],
    OTHER: PALETTE[6],
    TOTAL: NEUTRAL,
}


# --- Accès à Prometheus -----------------------------------------------------

class PrometheusError(Exception):
    """Prometheus injoignable, ou requête PromQL refusée."""


def _prom_get(path: str, params: dict | None = None) -> dict:
    url = f"{PROMETHEUS_URL}{path}"
    try:
        response = httpx.get(url, params=params, timeout=HTTP_TIMEOUT)
    except httpx.HTTPError as exc:
        raise PrometheusError(f"{type(exc).__name__} : {exc}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        status = response.status_code
        raise PrometheusError(f"réponse non JSON (HTTP {status})") from exc
    if payload.get("status") != "success":
        raise PrometheusError(
            f"{payload.get('errorType', f'HTTP {response.status_code}')} : "
            f"{payload.get('error', response.text[:200])}"
        )
    return payload["data"]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def prom_buildinfo() -> dict:
    return _prom_get("/api/v1/status/buildinfo")


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def prom_targets() -> list[dict]:
    return _prom_get("/api/v1/targets", {"state": "active"})["activeTargets"]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def prom_query(query: str) -> list[dict]:
    return _prom_get("/api/v1/query", {"query": query})["result"]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def prom_query_range(query: str, window_s: int, step_s: int) -> list[dict]:
    # `end` est calculé ici (et arrondi au pas) pour que le cache fonctionne
    # avec la fenêtre comme clé.
    end = int(time.time()) // step_s * step_s
    params = {"query": query, "start": end - window_s, "end": end, "step": step_s}
    return _prom_get("/api/v1/query_range", params)["result"]


def _clear_cache():
    for cached in (prom_buildinfo, prom_targets, prom_query, prom_query_range):
        cached.clear()


def _instant_value(query: str) -> float | None:
    result = prom_query(query)
    if not result:
        return None
    value = float(result[0]["value"][1])
    return value if np.isfinite(value) else None


# --- Mise en forme ----------------------------------------------------------

def _fmt_int(value: float | None) -> str:
    return f"{int(round(value or 0)):,}".replace(",", " ")  # espace fine insécable


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days} j {hours} h"
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def _handler_group(metric: dict) -> str:
    handler = metric.get("handler", OTHER)
    if handler.startswith("/train/simulation"):
        return SIMULATION
    return handler if handler in HANDLER_COLORS else OTHER


def _named(name: str):
    return lambda metric: name


def _prometheus_graph_url(queries: list[str], range_input: str) -> str:
    """Interface de Prometheus (v3) avec les requêtes du graphique pré-remplies."""
    params = {}
    for i, query in enumerate(queries):
        params |= {f"g{i}.expr": query, f"g{i}.tab": "graph",
                   f"g{i}.range_input": range_input}
    return f"{PUBLIC_PROMETHEUS_URL}/query?{urlencode(params)}"


def _result_to_df(result: list[dict], namer, scale: float) -> pd.DataFrame:
    rows = [
        (float(ts), namer(serie["metric"]), float(value) * scale)
        for serie in result
        for ts, value in serie["values"]
    ]
    df = pd.DataFrame(rows, columns=["ts", "series", "value"])
    # NaN (pas de trafic) et ±Inf ; astype car la colonne est "object" si aucune série
    df = df[np.isfinite(df["value"].astype(float))]
    # Plusieurs séries peuvent tomber dans le même groupe (ex. "autres")
    df = df.groupby(["ts", "series"], as_index=False)["value"].sum()
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df[["time", "series", "value"]]


# --- Graphiques -------------------------------------------------------------

def _line_chart(
    df: pd.DataFrame, *, colors: dict, y_title: str, value_format: str,
    dashed: tuple = (), zero: bool = True, log: bool = False,
) -> alt.Chart:
    """Courbes temporelles avec réticule et infobulle listant toutes les séries."""
    if log:
        df = df[df["value"] > 0]
    present = set(df["series"])
    domain = [name for name in colors if name in present]
    legend = (alt.Legend(title=None, orient="bottom", symbolStrokeWidth=3)
              if len(domain) > 1 else None)

    if log:
        y_scale = alt.Scale(type="log")
    elif df["value"].max() == 0:  # séries toutes nulles : axe 0 → 1 plutôt qu'écrasé
        y_scale = alt.Scale(domain=[0, 1])
    else:
        y_scale = alt.Scale(zero=zero)

    x = alt.X("time:T", title=None,
              axis=alt.Axis(format="%H:%M", grid=False, labelOverlap=True))
    y = alt.Y("value:Q", title=y_title, scale=y_scale)
    palette = [colors[name] for name in domain]
    color = alt.Color("series:N", scale=alt.Scale(domain=domain, range=palette),
                      legend=legend)
    dashes = [[6, 4] if name in dashed else [1, 0] for name in domain]
    dash = alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=dashes),
                          legend=legend)

    hover = alt.selection_point(
        fields=["time"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )
    when_hover = alt.when(hover)

    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=x, y=y, color=color, strokeDash=dash
    )
    points = alt.Chart(df).mark_circle(size=64).encode(
        x=x, y=y, color=color,
        opacity=when_hover.then(alt.value(1)).otherwise(alt.value(0)),
    )

    # Tableau large (une colonne par série, noms neutres c0, c1... pour éviter
    # que les "/" ou "." des noms d'endpoints soient interprétés par Vega)
    wide = df.pivot_table(index="time", columns="series", values="value", aggfunc="sum")
    wide = wide.reindex(columns=domain).reset_index()
    wide.columns = ["time"] + [f"c{i}" for i in range(len(domain))]
    tooltip = [alt.Tooltip("time:T", title="Heure", format="%H:%M:%S")] + [
        alt.Tooltip(f"c{i}:Q", title=name, format=value_format)
        for i, name in enumerate(domain)
    ]
    rule = alt.Chart(wide).mark_rule(color=NEUTRAL, strokeWidth=1).encode(
        x="time:T",
        opacity=when_hover.then(alt.value(0.7)).otherwise(alt.value(0)),
        tooltip=tooltip,
    ).add_params(hover)

    return alt.layer(lines, points, rule).properties(height=260)


def _chart_card(title: str, help_text: str, queries: list[tuple[str, object]], window,
                *, scale: float = 1.0, wide: bool = False, **chart_options):
    """Encadré : titre, bouton vers Prometheus, graphique (ou message si vide).

    `wide` : encadré pleine largeur (le bouton garde alors la même taille que
    dans les encadrés en demi-largeur). `chart_options` va à `_line_chart`.
    """
    window_s, step_s, range_label = window
    with st.container(border=True, height="stretch"):
        col_title, col_link = st.columns(
            [5, 1] if wide else [2.2, 1], vertical_alignment="center"
        )
        col_title.markdown(f"**{title}**", help=help_text)
        col_link.link_button(
            "Prometheus ↗",
            _prometheus_graph_url([query for query, _ in queries], range_label),
            width="stretch",
        )
        frames, failed = [], False
        for query, namer in queries:
            try:
                result = prom_query_range(query, window_s, step_s)
            except PrometheusError as exc:
                st.warning(f"Requête refusée ou en échec : `{query}` → {exc}")
                failed = True
                continue
            frame = _result_to_df(result, namer, scale)
            if not frame.empty:
                frames.append(frame)

        if frames:
            chart = _line_chart(pd.concat(frames, ignore_index=True), **chart_options)
            st.altair_chart(chart, width="stretch")
        elif not failed:
            st.info(
                "Aucune série renvoyée par Prometheus sur cette période "
                "(pas encore de trafic, ou Prometheus vient de démarrer)."
            )


def _show_tiles(window_label: str, window):
    rng = window[2]
    try:
        total = _instant_value("sum(http_requests_total)")
        total_window = _instant_value(f"sum(increase(http_requests_total[{rng}]))")
        predict = _instant_value('sum(http_requests_total{handler="/predict"})')
        predict_window = _instant_value(
            f'sum(increase(http_requests_total{{handler="/predict"}}[{rng}]))'
        )
        by_status = prom_query(
            'sum by (status) (http_requests_total{handler="/predict"})'
        )
        memory = _instant_value("sum(process_resident_memory_bytes)")
        memory_delta = _instant_value(
            "sum(process_resident_memory_bytes"
            f" - process_resident_memory_bytes offset {rng})"
        )
        uptime = _instant_value("max(time() - process_start_time_seconds)")
    except PrometheusError as exc:
        st.warning(f"Impossible de lire les valeurs instantanées : {exc}")
        return
    try:
        targets = prom_targets()
    except PrometheusError:
        targets = []

    def increase(value):
        return f"+{_fmt_int(value)} sur {window_label}" if value is not None else None

    col_total, col_predict, col_memory, col_scrape = st.columns(4)
    col_total.metric(
        "Requêtes depuis le démarrage",
        _fmt_int(total),
        delta=increase(total_window),
        delta_color="off",
        border=True,
        help="`sum(http_requests_total)` : toutes les routes, y compris `/health` "
             "(healthcheck docker) et `/metrics` (scrapes de Prometheus). "
             "Le compteur repart de zéro si l'API redémarre.",
    )
    statuses = sorted(
        (s["metric"].get("status", "?"), float(s["value"][1])) for s in by_status
    )
    status_detail = ", ".join(f"{code} : {_fmt_int(n)}" for code, n in statuses)
    col_predict.metric(
        "Requêtes /predict",
        _fmt_int(predict),
        delta=increase(predict_window),
        delta_color="off",
        border=True,
        help='`sum(http_requests_total{handler="/predict"})`'
             + (f" — par classe de status : {status_detail}" if status_detail else ""),
    )
    started = "" if uptime is None else f" API démarrée il y a {_fmt_duration(uptime)}."
    col_memory.metric(
        "Mémoire de l'API",
        f"{memory / 1e6:.0f} Mo" if memory is not None else "—",
        delta=(f"{memory_delta / 1e6:+.0f} Mo sur {window_label}"
               if memory_delta is not None else None),
        delta_color="off",
        border=True,
        help="`process_resident_memory_bytes` : mémoire résidente (RSS) du processus "
             "uvicorn de l'API." + started,
    )
    if targets:
        target = targets[0]
        healthy = sum(t["health"] == "up" for t in targets)
        value = ("🟢 UP" if healthy == len(targets)
                 else f"🔴 {healthy}/{len(targets)} UP")
        age = time.time() - datetime.fromisoformat(target["lastScrape"]).timestamp()
        delta = f"dernier scrape il y a {age:.0f} s"
        help_text = (
            f"Cible `{target['scrapeUrl']}` (job `{target['scrapePool']}`), "
            f"scrapée toutes les {target['scrapeInterval']} ; "
            f"durée du dernier scrape : {target['lastScrapeDuration'] * 1000:.0f} ms."
        )
        if target["lastError"]:
            help_text += f" Dernière erreur : {target['lastError']}"
    else:
        value, delta = "—", None
        help_text = "Aucune cible active déclarée dans Prometheus."
    col_scrape.metric("Collecte par Prometheus", value, delta=delta, delta_color="off",
                      border=True, help=help_text)


def _show_prometheus_charts(window):
    _chart_card(
        "Requêtes par seconde, totales et par endpoint",
        "Débit moyen sur une fenêtre glissante d'1 minute, toutes méthodes et tous "
        "status confondus. Les appels réguliers à `/health` (healthcheck, toutes les "
        "10 s) et à `/metrics` (Prometheus, toutes les 15 s) forment un bruit de fond "
        "constant.",
        [
            ("sum by (handler) (rate(http_requests_total[1m]))", _handler_group),
            ("sum(rate(http_requests_total[1m]))", _named(TOTAL)),
        ],
        window,
        colors=HANDLER_COLORS, y_title="requêtes / s", value_format=".3f",
        dashed=(TOTAL,), wide=True,
    )

    col_latency, col_errors = st.columns(2)
    with col_latency:
        quantiles = [("p50", 0.50), ("p95", 0.95), ("p99", 0.99)]
        _chart_card(
            "Latence des requêtes (p50 / p95 / p99)",
            "Quantiles estimés depuis l'histogramme haute résolution "
            "`http_request_duration_highr_seconds` (21 buckets, sans label "
            "`handler`), comme le dashboard ML. Fenêtre de 5 min pour lisser. "
            "Échelle logarithmique : les requêtes courantes durent quelques ms "
            "alors qu'un entraînement synchrone (`/train`) peut dépasser la minute. "
            "Au-delà du plus grand bucket fini (60 s), le quantile est plafonné "
            "à 60 000 ms.",
            [
                (
                    f"histogram_quantile({q}, sum by (le) "
                    f"(rate(http_request_duration_highr_seconds_bucket[5m])))",
                    _named(name),
                )
                for name, q in quantiles
            ],
            window,
            colors={name: PALETTE[i] for i, (name, _) in enumerate(quantiles)},
            y_title="latence (ms, échelle log)", value_format=",.1f", scale=1000,
            log=True,
        )
    with col_errors:
        _chart_card(
            "Erreurs par seconde",
            "Requêtes terminées en erreur serveur (5xx) ou client (4xx : clé d'API "
            "manquante, payload invalide...). L'instrumentator regroupe les status "
            "par classe (`2xx`, `4xx`, `5xx`). `or vector(0)` force une courbe à zéro "
            "quand il n'y a aucune erreur.",
            [
                ('sum(rate(http_requests_total{status=~"5.."}[1m])) or vector(0)',
                 _named("5xx")),
                ('sum(rate(http_requests_total{status=~"4.."}[1m])) or vector(0)',
                 _named("4xx")),
            ],
            window,
            colors={"5xx": PALETTE[7], "4xx": PALETTE[1]},
            y_title="erreurs / s", value_format=".3f",
        )

    col_cpu, col_memory = st.columns(2)
    with col_cpu:
        _chart_card(
            "CPU de l'API",
            "Secondes CPU consommées par seconde par le processus de l'API, en % d'un "
            "cœur (peut dépasser 100 % pendant un entraînement multi-thread).",
            [("rate(process_cpu_seconds_total[1m]) * 100", _named("CPU"))],
            window,
            colors={"CPU": PALETTE[0]}, y_title="% d'un cœur", value_format=".1f",
        )
    with col_memory:
        _chart_card(
            "Mémoire de l'API",
            "Mémoire résidente (RSS) du processus de l'API, en mégaoctets.",
            [("process_resident_memory_bytes / 1e6", _named("Mémoire"))],
            window,
            colors={"Mémoire": PALETTE[0]}, y_title="Mo", value_format=".1f",
            zero=False,
        )


# --- Grafana ----------------------------------------------------------------

GRAFANA_DASHBOARD_DIR = REPO_ROOT / "grafana" / "provisioning" / "dashboards"
GRAFANA_PANEL_HEIGHT = 340  # px, identique pour tous les panels -> grille alignée
GRAFANA_PANELS_PER_ROW = 2


@st.cache_data(ttl=300, show_spinner=False)
def _dashboard_panel_ids(uid: str) -> list[int]:
    """Ids des panels d'un dashboard provisionné, dans l'ordre de sa grille.

    Liste vide si le dashboard est introuvable ou de structure inattendue.
    """
    for path in GRAFANA_DASHBOARD_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["metadata"]["name"] != uid:
                continue
            spec = data["spec"]
            # GridLayout : tri par (y, x) ; AutoGridLayout : sans position, ordre gardé
            items = sorted(
                spec["layout"]["spec"]["items"],
                key=lambda item: (item["spec"].get("y", 0), item["spec"].get("x", 0)),
            )
            return [spec["elements"][item["spec"]["element"]["name"]]["spec"]["id"]
                    for item in items]
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return []


def _show_grafana(grafana_range: str):
    col_title, col_link = st.columns([5, 1.3], vertical_alignment="center")
    col_title.subheader("📊 Dashboards Grafana")
    col_link.link_button("Ouvrir Grafana ↗", PUBLIC_GRAFANA_URL, width="stretch")

    time_range = f"from=now-{grafana_range}&to=now&refresh=10s"
    tabs = st.tabs(list(GRAFANA_DASHBOARDS.values()))
    for tab, uid in zip(tabs, GRAFANA_DASHBOARDS, strict=True):
        panel_ids = _dashboard_panel_ids(uid)
        if not panel_ids:  # repli : le dashboard entier
            tab.iframe(f"{PUBLIC_GRAFANA_URL}/d/{uid}?orgId=1&kiosk&{time_range}",
                       height=700)
            continue
        # Un panel par iframe, en grille de hauteur fixe : pas de barre de défilement
        for start in range(0, len(panel_ids), GRAFANA_PANELS_PER_ROW):
            row = panel_ids[start:start + GRAFANA_PANELS_PER_ROW]
            cols = tab.columns(GRAFANA_PANELS_PER_ROW)
            for col, panel_id in zip(cols, row, strict=False):
                col.iframe(
                    f"{PUBLIC_GRAFANA_URL}/d-solo/{uid}?orgId=1&panelId={panel_id}"
                    f"&{time_range}",
                    height=GRAFANA_PANEL_HEIGHT,
                )


# --- Explications -----------------------------------------------------------

def _show_how_it_works():
    st.subheader("⚙️ Comment ça marche")
    try:
        config = yaml.safe_load(PROMETHEUS_CONFIG_FILE.read_text(encoding="utf-8"))
        job = config["scrape_configs"][0]
        target = job["static_configs"][0]["targets"][0]
        default_interval = config.get("global", {}).get("scrape_interval", "1m")
        interval = job.get("scrape_interval", default_interval)
        scrape = f"lit `{target}` toutes les **{interval}**"
    except (OSError, yaml.YAMLError, LookupError, TypeError):
        scrape = "lit l'API à intervalle régulier (`prometheus.yml`)"
    steps = [
        ("1. Exposition", "FastAPI publie ses métriques sur `/metrics` "
                          "(`prometheus-fastapi-instrumentator`)."),
        ("2. Collecte", f"Prometheus {scrape} et stocke les séries, "
                        "interrogeables en PromQL."),
        ("3. Visualisation", "Grafana lit Prometheus ; ses dashboards sont "
                             "provisionnés depuis `grafana/provisioning/`."),
    ]
    for col, (title, body) in zip(st.columns(3), steps, strict=True):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.markdown(body)


# --- Page -------------------------------------------------------------------

def show_monitoring():
    st.header("📡 Monitoring")
    st.markdown(
        "Suivi en temps réel de l'API d'inférence : l'API expose ses métriques, "
        "**Prometheus** les collecte et **Grafana** les met en forme. Les graphiques "
        "ci-dessous interrogent directement Prometheus."
    )

    col_window, col_refresh, col_info = st.columns([1, 1, 3],
                                                   vertical_alignment="bottom")
    window_label = col_window.selectbox(
        "Fenêtre", list(WINDOWS), index=list(WINDOWS).index(DEFAULT_WINDOW),
        key="monitoring_window",
    )
    col_refresh.button("🔄 Rafraîchir", on_click=_clear_cache, key="monitoring_refresh")
    window = WINDOWS[window_label]

    try:
        build = prom_buildinfo()
    except PrometheusError as exc:
        st.error(
            f"Prometheus est injoignable à l'adresse `{PROMETHEUS_URL}` ({exc}). "
            "Vérifier que le service tourne (`docker compose ps prometheus`)."
        )
    else:
        col_info.caption(
            f"Prometheus {build.get('version', '?')} · affiché à "
            f"{datetime.now():%H:%M:%S} · données mises en cache {CACHE_TTL} s "
            "(« Rafraîchir » pour forcer la mise à jour)"
        )
        _show_tiles(window_label, window)
        _show_prometheus_charts(window)

    st.divider()
    _show_grafana(window[2])
    st.divider()
    _show_how_it_works()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_monitoring()
