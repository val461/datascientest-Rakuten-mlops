"""Page « Expériences MLflow » : suivi des entraînements journalisés dans MLflow.

Les runs sont produits par ``src/trainer.py`` (``log_training_run``) :
- ``POST /train`` → run ``LinearSVC-full-training`` ;
- campagne de simulation → runs ``LinearSVC-step-XX-ratio-R.RR``
  (params ``simulation_step``, ``available_ratio``…, tag ``simulation=data-growth``).
"""

import altair as alt
import httpx
import mlflow
import numpy as np
import pandas as pd
import streamlit as st

try:
    from parts import config
except ImportError:  # streamlit run src/streamlit/parts/experiments.py
    import config

N_SIMULATION_STEPS = config.MAX_SIMULATION_STEP + 1  # steps 0 à 10

# clé de la métrique MLflow -> libellé affiché
METRICS = {
    "accuracy": "Accuracy",
    "f1_macro": "F1 macro",
    "f1_weighted": "F1 pondéré",
}

KIND_SIMULATION = "Simulation"
KIND_FULL = "Complet (POST /train)"
KIND_OTHER = "Autre"

# 3 premières couleurs catégorielles (palette validée daltonisme, toutes paires)
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70"]

CURL_POST = """curl -X 'POST' \\
  '{url}' \\
  -H 'accept: application/json' \\
  -H 'X-API-Key: your_api_key' \\
  -d ''"""


# --- Récupération et préparation des données ------------------------------

@st.cache_data(ttl=30, show_spinner="Interrogation de MLflow…")
def fetch_runs() -> tuple[str | None, pd.DataFrame]:
    """(experiment_id, runs bruts) ; experiment_id vaut None si l'expérience
    n'existe pas encore. Une exception (MLflow injoignable) n'est pas mise en cache.
    """
    # Test rapide : le client mlflow réessaie longtemps si le serveur est absent.
    httpx.get(
        f"{config.MLFLOW_URL}/health", timeout=config.HTTP_TIMEOUT
    ).raise_for_status()

    mlflow.set_tracking_uri(config.MLFLOW_URL)
    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return None, pd.DataFrame()
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )
    return experiment.experiment_id, runs


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """Colonne `name` de df, ou colonne vide si MLflow ne l'a pas renvoyée."""
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _numeric(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_column(df, name), errors="coerce")


def prepare_runs(raw: pd.DataFrame, exp_url: str) -> pd.DataFrame:
    step = _numeric(raw, "params.simulation_step")
    run_name = _column(raw, "tags.mlflow.runName").fillna("").astype(str)
    is_simulation = step.notna() | (
        _column(raw, "tags.simulation").fillna("") == "data-growth"
    )
    is_full = ~is_simulation & run_name.str.endswith("-full-training")
    # L'ordre des colonnes est celui du tableau « Tous les runs ».
    return pd.DataFrame(
        {
            "run_name": run_name,
            "kind": np.select(
                [is_simulation, is_full], [KIND_SIMULATION, KIND_FULL], KIND_OTHER
            ),
            "start_time": raw["start_time"],
            "status": raw["status"],
            "step": step.astype("Int64"),
            "available_ratio_pct": _numeric(raw, "params.available_ratio") * 100,
            "train_rows": _numeric(raw, "params.train_rows").astype("Int64"),
            **{key: _numeric(raw, f"metrics.{key}") for key in METRICS},
            "url": exp_url + "/runs/" + raw["run_id"],
        }
    )


# --- Graphique ------------------------------------------------------------

def build_growth_chart(points: pd.DataFrame) -> alt.LayerChart:
    wide = pd.DataFrame(
        {
            "ratio": points["available_ratio_pct"],
            "step": points["step"].astype(int),
            "train_rows": points["train_rows"].astype(float),
            "run_name": points["run_name"],
            **{key: points[key] for key in METRICS},
        }
    )
    long = wide.melt(
        id_vars=["ratio", "step", "train_rows", "run_name"],
        value_vars=list(METRICS),
        var_name="metric_key",
        value_name="score",
    )
    long["metric"] = long["metric_key"].map(METRICS)

    # Axe Y zoomé sur les données, pas forcé à 0.
    low, high = float(long["score"].min()), float(long["score"].max())
    pad = max((high - low) * 0.15, 0.005)

    x = alt.X(
        "ratio:Q",
        title="Part du flux disponible pour l'entraînement (%)",
        scale=alt.Scale(domain=[46, 105], nice=False),
        axis=alt.Axis(values=list(range(50, 101, 5)), grid=False),
    )
    y = alt.Y(
        "score:Q",
        title="Score sur le bloc de validation fixe",
        scale=alt.Scale(domain=[low - pad, high + pad], zero=False, nice=False),
        axis=alt.Axis(format=".3f"),
    )
    palette = PALETTE_DARK if st.context.theme.type == "dark" else PALETTE_LIGHT
    metric_scale = alt.Scale(domain=list(METRICS.values()), range=palette)
    # Légende au-dessus de la zone tracée, pour ne jamais chevaucher les courbes.
    legend = alt.Legend(title=None, orient="top", direction="horizontal")

    base = alt.Chart(long).encode(
        x=x, y=y, color=alt.Color("metric:N", scale=metric_scale, legend=legend)
    )
    lines = base.mark_line(strokeWidth=2)
    markers = base.mark_point(filled=True, size=90, opacity=1).encode(
        shape=alt.Shape("metric:N", legend=legend)
    )

    # Survol : bande invisible autour de chaque étape -> ligne verticale
    # + infobulle avec les 3 métriques de l'étape.
    hover = alt.selection_point(
        fields=["ratio"], on="pointerover", empty=False, clear="pointerout"
    )
    tooltip = [
        alt.Tooltip("step:Q", title="Étape"),
        alt.Tooltip("ratio:Q", title="Part du flux (%)", format=".0f"),
        alt.Tooltip("train_rows:Q", title="Lignes d'entraînement", format=",.0f"),
        *[
            alt.Tooltip(f"{key}:Q", title=label, format=".4f")
            for key, label in METRICS.items()
        ],
        alt.Tooltip("run_name:N", title="Run"),
    ]
    hover_rule = (
        alt.Chart(wide)
        .mark_rule(color="gray", strokeWidth=1)
        .encode(
            x=x, opacity=alt.when(hover).then(alt.value(0.6)).otherwise(alt.value(0))
        )
    )
    hover_targets = (
        alt.Chart(wide)
        .mark_rule(strokeWidth=50, opacity=0)
        .encode(x=x, tooltip=tooltip)
        .add_params(hover)
    )

    # Valeurs affichées au premier et au dernier step (accuracy et F1 macro :
    # F1 pondéré est quasi confondu avec l'accuracy, son étiquette se superposerait).
    first_ratio, last_ratio = long["ratio"].min(), long["ratio"].max()
    labelled = long[long["metric_key"].isin(["accuracy", "f1_macro"])]
    end_labels = [
        alt.Chart(labelled[labelled["ratio"] == ratio])
        .mark_text(align=align, dx=dx, fontSize=11)
        .encode(
            x=x,
            y=y,
            text=alt.Text("score:Q", format=".3f"),
            color=alt.Color("metric:N", scale=metric_scale, legend=None),
        )
        for ratio, align, dx in [(first_ratio, "right", -10), (last_ratio, "left", 10)]
    ]

    layers = [lines, markers, hover_rule]
    if first_ratio != last_ratio:
        layers += end_labels
    return alt.layer(*layers, hover_targets).properties(height=420)


# --- Mise en forme --------------------------------------------------------

def _fmt_int(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(value):,}".replace(",", " ")  # espace fine insécable


def _fmt_time(timestamp: pd.Timestamp) -> str:
    try:
        timezone = st.context.timezone
        if timezone:
            return timestamp.tz_convert(timezone).strftime("%d/%m/%Y %H:%M")
    except Exception:
        pass
    return timestamp.strftime("%d/%m/%Y %H:%M UTC")


def _pts(delta: float) -> str:
    """Écart de score en points de pourcentage."""
    return f"{delta * 100:+.2f} pts"


# --- Sections -------------------------------------------------------------

def _show_launch_help() -> None:
    st.markdown(
        "Lancer un entraînement complet (≈ 10 min, crée un run "
        "`LinearSVC-full-training` et déploie le modèle) — remplacez "
        "`your_api_key` par la clé de votre `.env` :"
    )
    st.code(CURL_POST.format(url=f"{config.PUBLIC_API_URL}/train"), language="bash")
    st.markdown(
        f"Ou lancer la campagne de simulation ({N_SIMULATION_STEPS} étapes en "
        "arrière-plan, un run `LinearSVC-step-XX-ratio-R.RR` par étape ; "
        "avancement via `GET /train/simulation/status`) :"
    )
    st.code(
        CURL_POST.format(url=f"{config.PUBLIC_API_URL}/train/simulation"),
        language="bash",
    )


def _show_growth_section(runs: pd.DataFrame) -> None:
    st.subheader("Performance en fonction des données disponibles")
    st.caption(
        "Chaque step réentraîne le modèle sur une part croissante du flux "
        "(step 0 = 50 %, +5 % par step, step 10 = 100 %) et l'évalue sur le même "
        "bloc de validation fixe (20 % du dataset) : les points sont directement "
        "comparables."
    )

    finished = runs[
        (runs["status"] == "FINISHED") & runs[list(METRICS)].notna().all(axis=1)
    ]
    # fetch_runs trie du plus récent au plus ancien : pour une étape rejouée,
    # keep="first" garde le run le plus récent.
    points = (
        finished[(finished["kind"] == KIND_SIMULATION) & finished["step"].notna()]
        .drop_duplicates("step", keep="first")
        .sort_values("step", ignore_index=True)
    )
    n_steps = len(points)

    if n_steps == 0:
        st.info(
            "Aucune étape de simulation loguée pour l'instant "
            f"(0/{N_SIMULATION_STEPS}). Si la campagne vient d'être lancée, la "
            "première étape apparaîtra dans quelques minutes : cliquez sur "
            "« Rafraîchir »."
        )
        return

    if n_steps < N_SIMULATION_STEPS:
        st.info(
            f"La campagne est en cours : {n_steps}/{N_SIMULATION_STEPS} étapes "
            "loguées. Cliquez sur « Rafraîchir » pour voir les nouvelles étapes.",
            icon="⏳",
        )

    col_chart, col_kpi = st.columns([3, 1], gap="medium")
    with col_chart:
        st.altair_chart(build_growth_chart(points), width="stretch")
        note = (
            f"{n_steps} étape(s) tracée(s) · survolez une étape pour le détail "
            "des trois scores."
        )
        if (runs["kind"] == KIND_SIMULATION).sum() > n_steps:
            note += " Pour une étape rejouée, seul le run le plus récent est tracé."
        st.caption(note)
    with col_kpi:
        _show_growth_tiles(points, finished[finished["kind"] == KIND_FULL])


def _show_growth_tiles(points: pd.DataFrame, full_runs: pd.DataFrame) -> None:
    first, last = points.iloc[0], points.iloc[-1]
    first_step, last_step = int(first["step"]), int(last["step"])
    st.metric(
        f"Accuracy · step {first_step} ({first['available_ratio_pct']:.0f} %)",
        f"{first['accuracy']:.4f}",
        help=(
            f"{first['available_ratio_pct']:.0f} % du flux = "
            f"{_fmt_int(first['train_rows'])} lignes d'entraînement"
        ),
        border=True,
    )
    if len(points) >= 2:
        st.metric(
            f"Accuracy · step {last_step} ({last['available_ratio_pct']:.0f} %)",
            f"{last['accuracy']:.4f}",
            delta=_pts(last["accuracy"] - first["accuracy"]),
            help=(
                f"{last['available_ratio_pct']:.0f} % du flux = "
                f"{_fmt_int(last['train_rows'])} lignes d'entraînement"
                f" · écart vs step {first_step}"
            ),
            border=True,
        )
        st.metric(
            f"F1 macro · step {last_step}",
            f"{last['f1_macro']:.4f}",
            delta=_pts(last["f1_macro"] - first["f1_macro"]),
            help=f"Écart vs step {first_step} ({first['f1_macro']:.4f})",
            border=True,
        )

    if full_runs.empty:
        st.metric(
            "Référence POST /train",
            "—",
            help="Aucun run complet `LinearSVC-full-training` : lancer `POST /train`.",
            border=True,
        )
        return
    reference = full_runs.iloc[0]
    diff = reference["accuracy"] - last["accuracy"]
    same = abs(diff) < 0.0005
    st.metric(
        "Référence POST /train",
        f"{reference['accuracy']:.4f}",
        delta=None if same else f"{_pts(diff)} vs step {last_step}",
        delta_color="off",
        help=(
            f"Accuracy du run `{reference['run_name']}` du "
            f"{_fmt_time(reference['start_time'])}, "
            f"{_fmt_int(reference['train_rows'])} lignes (= 100 % du flux), "
            f"F1 macro {reference['f1_macro']:.4f}."
            + (f" Identique au step {last_step}." if same else "")
        ),
        border=True,
    )
    st.caption(f"[Ouvrir le run de référence ↗]({reference['url']})")


def _show_runs_table(runs: pd.DataFrame) -> None:
    st.subheader("Tous les runs")
    st.caption(f"{len(runs)} run(s) dans l'expérience, du plus récent au plus ancien.")
    st.dataframe(
        runs,
        hide_index=True,
        column_config={
            "run_name": st.column_config.TextColumn("Run"),
            "kind": st.column_config.TextColumn("Type"),
            "start_time": st.column_config.DatetimeColumn(
                "Début",
                format="DD/MM/YYYY HH:mm:ss",
                timezone=st.context.timezone or None,
            ),
            "status": st.column_config.TextColumn("Statut"),
            "step": st.column_config.NumberColumn("Étape", format="%d"),
            "available_ratio_pct": st.column_config.NumberColumn(
                "Part du flux", format="%.0f %%"
            ),
            "train_rows": st.column_config.NumberColumn(
                "Lignes d'entraînement", format="localized"
            ),
            **{
                key: st.column_config.NumberColumn(label, format="%.4f")
                for key, label in METRICS.items()
            },
            "url": st.column_config.LinkColumn("MLflow", display_text="Ouvrir ↗"),
        },
    )


# --- Page -----------------------------------------------------------------

def show_experiments():
    col_title, col_refresh, col_link = st.columns(
        [5, 1, 1.5], vertical_alignment="bottom"
    )
    col_title.title("📈 Expériences MLflow")
    if col_refresh.button("🔄 Rafraîchir", width="stretch"):
        fetch_runs.clear()

    try:
        experiment_id, raw = fetch_runs()
    except Exception as exc:
        st.error(
            f"MLflow est injoignable à l'adresse `{config.MLFLOW_URL}` : {exc}\n\n"
            "Vérifiez que le service `mlflow` tourne "
            f"(UI publique : {config.PUBLIC_MLFLOW_URL})."
        )
        return

    exp_url = config.PUBLIC_MLFLOW_URL
    if experiment_id is not None:
        exp_url += f"/#/experiments/{experiment_id}"
    col_link.link_button("Ouvrir dans MLflow ↗", exp_url, width="stretch")

    if raw.empty:
        st.info(
            f"L'expérience `{config.MLFLOW_EXPERIMENT_NAME}` ne contient encore "
            "aucun run. Lancez un entraînement pour la remplir."
        )
        _show_launch_help()
        return

    runs = prepare_runs(raw, exp_url)
    _show_growth_section(runs)
    st.divider()
    _show_runs_table(runs)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_experiments()
