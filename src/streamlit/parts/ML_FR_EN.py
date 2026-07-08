# parts/ML_FR_EN.py

import streamlit as st


def show_ML_FR_EN():
    st.title("Machine Learning en français et anglais \U0001F1EB\U0001F1F7 \U0001F1EC\U0001F1E7")
    st.title("📊 Visualisation des Données")

    st.markdown("**Visualisation du Dataframe en français et en anglais**")
    st.image(r"src/streamlit/images/sample_X_fr_rakuten.png")
    st.markdown("Ceci est un sample du dataframe avec uniquement les entrées en Français \U0001F1EB\U0001F1F7.(grâce à la librairie lang.detect)")
    st.image(r"src/streamlit/images/sample_X_en_rakuten.png")
    st.markdown("Ceci est un sample du dataframe avec uniquement les entrées en anglais \U0001F1EC\U0001F1E7. (grâce à la librairie lang.detect)")

    st.title("Étape 1 : entrainement sur le texte en français.")
    st.title("Cross-validation et choix du modèle")
    st.markdown("""
**Pour savoir quel modèle est le plus adapté pour nos jeux de données, on effectue une cross-validation avec optimisation des hyperparamètres.**

Étant donné le fort déséquilibre des classes, nous calculons des **poids de classe optimisés** (plafond à 5.0).

**Trois modèles comparés :**
- **LinearSVC** : SVM linéaire pour haute dimension
- **LogisticRegression** : modèle probabiliste (solver `saga`)
- **SGDClassifier** : descente de gradient avec early stopping

**Optimisation :** GridSearchCV avec 3 folds stratifiés, métrique **F1-score macro** (adapté au déséquilibre).

**Évaluation finale :** Accuracy, F1 Macro, F1 Weighted sur le test set.
""")
    st.image(r"src/streamlit/images/stats ml 1.png")
    st.image(r"src/streamlit/images/stats ml 2.png")


    st.markdown("Le meilleur modèle est SGDClassifier, dont voici la matrice de confusion et quelques graphiques à propos de ses résultats :")

    st.image(r"src/streamlit/images/stats ml 3.png")
    st.image(r"src/streamlit/images/stats ml 5.png")
    st.image(r"src/streamlit/images/stats ml 4.png")

    st.title("Étape 2 : entrainement sur le texte en anglais.")
    st.title("Cross-validation et choix du modèle")
    st.markdown("""
**Pour savoir quel modèle est le plus adapté pour nos jeux de données, on effectue une cross-validation avec optimisation des hyperparamètres.**

Étant donné le fort déséquilibre des classes, nous calculons des **poids de classe optimisés** (plafond à 5.0).

**Trois modèles comparés :**
- **LinearSVC** : SVM linéaire pour haute dimension
- **LogisticRegression** : modèle probabiliste (solver `saga`)
- **SGDClassifier** : descente de gradient avec early stopping

**Optimisation :** GridSearchCV avec 3 folds stratifiés, métrique **F1-score macro** (adapté au déséquilibre).

**Évaluation finale :** Accuracy, F1 Macro, F1 Weighted sur le test set.
""")
    st.image(r"src/streamlit/images/stats ml 1 en.png")
    st.image(r"src/streamlit/images/stats ml 2 en.png")


    st.markdown("Le meilleur modèle est LinearSVC, dont voici la matrice de confusion et quelques graphiques à propos de ses résultats :")

    st.image(r"src/streamlit/images/stats ml 3 en.png")
    st.image(r"src/streamlit/images/stats ml 5 en.png")
    st.image(r"src/streamlit/images/stats ml 4 en.png")


    # st.markdown("""
    # - liste markdown
    # - item
    # """)

    # Screenshot
    # st.image("screenshots/contexte.png", caption="Vue d'ensemble")

    # Tableau
    # st.header("Header")
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.metric("Catégories", "27")
    # with col2:
    #     st.metric("Articles", "~100K")


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_ML_FR_EN()
