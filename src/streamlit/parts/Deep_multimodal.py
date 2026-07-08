# parts/Deep_multimodal.py

import streamlit as st


def show_Deep_multimodal():
    st.title("Deep multimodal")
    st.markdown("Cette section détaille les résultats d'une experimentation de transfert learning multimodal avec Pytorch. Les caractèristiques techniques sont les suivantes :")
    st.title("MODÈLES PRÉ-ENTRAÎNÉS (Transfer Learning)")
    st.markdown("""
**ResNet50 (Image)**

**Source** : Torchvision (PyTorch)

**Pré-entraîné sur** : ImageNet (1.2M images, 1000 classes)

**Architecture** : 50 couches profondes (Residual Network)

**Poids** : ~98 MB

**Utilisation** : Extraction de features visuelles (2048 dimensions)
""")

    st.markdown("""
**BERT Tokenizer (Texte)**

**Source** : HuggingFace Transformers

**Modèle** : BERT Multilingue (104 langues)
**Vocabulaire** : ~119,547 tokens
**Utilisation** : Tokenisation du texte (convertit mots → indices))
""")
    st.title("ARCHITECTURES NEURONALES")
    st.markdown("""
**Branche Image (CNN)**

**Type** : Convolutional Neural Network

**Entrée** : Images 128×128×3 (RGB)

**Sortie** : Vecteur de 2048 features

**Branche Texte (Embedding)**

**Type** : Embedding Layer + Average Pooling

**Entrée** : Séquence d'indices de tokens

**Sortie* : Vecteur de 128 dimensions (moyenne des embeddings)

**Classifieur de Fusion (MLP)**

**Type : Multi-Layer Perceptron**


**Entrée** : Concaténation image (2048) + texte (128) = 2176

**Sortie** : Probabilités par classe)
""")

    st.title("TECHNIQUES D'OPTIMISATION")
    st.markdown("""
**Optimiseur**

**Algorithme** : Adam (Adaptive Moment Estimation)

**Learning Rate** : 0.0001

**Usage** : Met à jour les poids pour minimiser la perte

**Fonction de Perte**

**Type** : Cross-Entropy (classification multiclasse)

**Usage** : Mesure l'écart entre prédictions et vraies classes

**Mixed Precision Training**

**Technique** : Automatic Mixed Precision (AMP)

**Usage** : Utilise float16 au lieu de float32

**Bénéfice** : 2× plus rapide + économie de VRAM

**Régularisation**

**Technique** : Dropout (désactive 30% des neurones aléatoirement)

**Usage** : Prévient le surapprentissage (overfitting))
""")
    st.title("PREPROCESSING & DATA AUGMENTATION")
    st.markdown("""
**Transformations d'Images**

- Redimensionnement** : 128×128 pixels

- Conversion en tenseur [0,1]
- Normalisation ImageNet (mean/std standardisés)

**Source** : Torchvision

**Tokenisation de Texte**

- Troncature à 64 tokens maximum

- Padding automatique

- Conversion texte → indices numériques

**Source** : HuggingFace Transformers
""")

    st.title("GESTION DES DONNÉES")
    st.markdown("""
**Dataset PyTorch**

**Type** : Custom Dataset multimodal

**Usage** : Charge images + texte de façon lazy (à la demande)

**DataLoader**

**Mini-batches** : 64 exemples par batch

**Shuffle** : Mélange les données à chaque epoch

**Usage** : Pipeline d'entraînement efficace

**Train/Val Split**

**Source** : Scikit-learn

**Répartition** : 90% train / 10% validation

**Stratification** : Préserve la distribution des classes)
""")

    st.title("Résultats après entrainement :")
    st.image(r"src/streamlit/images/rapport de classification DL.png")
    st.image(r"src/streamlit/images/confusion matrix dl.png")
    st.markdown("""
**Résumé des performances du modèle :**

- **Accuracy globale** : 66% sur 8 492 échantillons de test
- **F1-Score macro** : 0.60 (moyenne non pondérée par classe)
- **F1-Score weighted** : 0.65 (pondéré par la taille des classes)

**✅ Points forts - Classes excellemment prédites :**
- Classe 2905 : F1=0.94 (precision 93%, recall 95%)
- Classe 1160 : F1=0.91 (precision 91%, recall 90%)
- Classe 2705 : F1=0.87 (precision 79%, recall 95%)
- Classe 2583 : F1=0.83 (precision 79%, recall 88%)

**⚠️ Points faibles - Classes problématiques :**
- Classes 1281, 2220, 1180 : F1 < 0.38, avec faible recall

Le modèle démontre d'**excellentes performances** sur plusieurs classes majeures, mais reste hétérogène sur les catégories minoritaires.
""")
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
    show_Deep_multimodal()
