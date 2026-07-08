# présentation du sujet, du problème et des enjeux

import streamlit as st


def show_problematique():
    # exemple fourni par le LLM Claude, à vérifier / retravailler
    st.title("🎯 Problématique")
    st.info("🚧 Section en cours de développement")

    st.header("Contexte")
    st.markdown("""
    - Rakuten : marketplace japonaise
    - Problème : catégorisation automatique de produits
    - Données multi-modales : texte + images
    """)

    # Screenshot
    # st.image("screenshots/contexte.png", caption="Vue d'ensemble")

    st.header("Enjeux métier")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Catégories", "27")
    with col2:
        st.metric("Articles", "~100K")


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_problematique()
