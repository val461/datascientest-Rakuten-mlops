# parts/mon_titre.py

import streamlit as st


def show_mon_titre():
    st.title("Mon titre")
    st.header("🚧 Section en cours de développement")
    # st.markdown("""
    # - liste markdown
    # - item
    # """)

    # Screenshot
    # st.image("screenshots/contexte.png", caption="Vue d'ensemble")


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_mon_titre()
