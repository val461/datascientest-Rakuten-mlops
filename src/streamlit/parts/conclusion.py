# conclusion du projet en reliant au maximum les résultats obtenus à la problématique métier
# critique et perspectives (ce qui aurait pû être fait avec plus de temps)
# par exemple : on aurait pu lire des articles sur des sujets similaires pour s'inspirer de leurs architectures

import streamlit as st


def show_conclusion():
    st.title("🎓 Conclusion & perspectives")
    st.header("🚧 Section en cours de développement")
    # exemple fourni par le LLM Claude, à vérifier / retravailler

    st.header("Résultats obtenus")
    # st.image("screenshots/resultats_finaux.png")

    st.header("Lien avec la problématique métier")
    st.markdown("""
    - Amélioration de X% de la catégorisation
    - Réduction du temps de traitement
    - Impact sur l'expérience utilisateur
    """)

    # Tableau
    st.header("Critiques & Perspectives")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ Limites")
        st.markdown("- Temps de calcul\n- Déséquilibre des classes")
    with col2:
        st.subheader("🚀 Améliorations")
        st.markdown("- Augmentation de données\n- Ensemble de modèles")


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_conclusion()
