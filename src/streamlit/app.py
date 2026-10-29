# Pour lancer cette application :
# se placer dans le dossier du repo git, puis lancer les commandes suivantes.
# source venv/bin/activate
# streamlit run src/streamlit/app.py

import streamlit as st
import sys
from pathlib import Path

# Quick fix for import issues
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

st.set_page_config(
    page_title="Projet Rakuten - MLOps",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reduce sidebar width to gain space for demo
# st.markdown(
#     """
#     <style>
#         section[data-testid="stSidebar"] {
#             width: 300px !important; # Set the width to your desired value
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Import de toutes les fonctions show_*
# Une fonction show_* par fichier de préférence, pour faciliter la collaboration
# Les pages data science de l'ancienne présentation restent dans parts/drafts/
# (importables avec `from parts.drafts.<nom> import show_<nom>`).
from parts.architecture import show_architecture
from parts.api import show_api
from parts.experiments import show_experiments
from parts.orchestration import show_orchestration
from parts.monitoring import show_monitoring

# Handle the possibility that loading demo fails because of missing dependencies (tensorflow, ...)
demo_available = True
try:
    from parts.demo import show_demo
except ImportError as e:
    demo_available = False
    exc = e  # strangely, `e` is not accessible from `show_demo` but `exc` is
    def show_demo():
        st.error(f"🚧 Démo non disponible :\n\n{exc}")

# Configuration de la navigation : partie → fonction montrant la partie
PAGES = {
    "🏗️ Architecture": show_architecture,
    "🚀 Démo interactive": show_demo,
    "🔌 API FastAPI": show_api,
    "📈 Expériences MLflow": show_experiments,
    "⏱️ Orchestration Airflow": show_orchestration,
    "📡 Monitoring": show_monitoring,
}

st.sidebar.title("Projet Rakuten")

# Navigation
page = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[page]()
