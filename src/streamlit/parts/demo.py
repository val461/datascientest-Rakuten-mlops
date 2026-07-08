import os
import streamlit as st
import base64
import httpx
from pathlib import Path
from functools import partial

from src.data_loader import load_training_csv

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")


@st.cache_data
def get_category_mapping():
    category_mapping = {}
    raw_mapping_text="""2583 équipements de piscine
    1560 meubles, chaises, matelas, bibelots
    1300 modélisme, drones, caméras type go-pro
    2060 décorations d'intérieur, souvent lumineuses
    2522 papeterie
    1280 peluches
    2403 livres, BD, mangas (souvent occasion/anciens)
    2280 journaux, magazines, livres documentaires
    1920 objets en tissu, linge de maison
    1160 cartes à jouer et à collectionner
    1320 accessoires pratiques pour bébés
    10 livres (poche, romans)
    2705 livres (beaux livres, art)
    1140 figurines pour enfants
    2582 accessoires de jardin
    40 jeux-vidéo 'rétro' et accessoires
    2585 accessoires pour la maison (bricolage/outils)
    1302 accessoires sportifs et voyage enfants
    1281 jeux de société pour enfants
    50 accessoires gaming, câbles
    2462 jeux vidéo, consoles, accessoires
    2905 jeux vidéo PC (boite/téléchargement)
    60 consoles de jeux-vidéo
    2220 accessoires pour animaux
    1301 jeux/accessoires nouveaux-nés
    1940 nourriture (conserve/sous vide)
    1180 jeux de rôle, plateau, figurines"""

    for line in raw_mapping_text.strip().split('\n'):
        if line.strip():
            # .split(' ', 1) splits only on the FIRST space.
            # The first part is the ID, the rest is the description.
            code_str, description = line.strip().split(' ', 1)

            # We store the key as INT to match LabelEncoder classes usually
            # If your classes are strings, remove int()
            category_mapping[int(code_str)] = description
    return category_mapping


@st.cache_data
def load_dataset():
    X, y = load_training_csv()
    return X, y


def np_array_to_int(np_array):
    return int(np_array[0])


def predict(row_df):
    row_dict = row_df.to_dict('records')[0]
    response = httpx.post(f"{BASE_URL}/predict", headers={"X-API-Key": API_KEY}, json=row_dict)
    data = response.json()
    y_pred_class = data.get("prediction")
    return y_pred_class


@st.cache_data
def get_class_description(class_code: int):
    CATEGORY_MAPPING = get_category_mapping()
    description = CATEGORY_MAPPING.get(class_code, "inconnue")
    return description


def get_image_path(row, folder = 'Dataset/images/image_train', as_string=False):
    filename = f"image_{row.imageid}_product_{row.productid}.jpg"
    path = Path(folder) / filename
    if as_string:
        path = str(path)
    return path


def show_image_from_row(row):
    # e.g. row = X_test.iloc[0]
    image_path = get_image_path(row, folder = 'Dataset/images/image_train')
    if image_path.exists():
        st.image(image_path)
    else:
        st.text(f"Error: file not found: {image_path=}")
        st.text(f"cwd: {Path('.').resolve()}")
        st.text(f"folder: {image_path.parent.resolve()}")


def image_path_to_base64(path: str):
    with open(path, "rb") as p:
        file = p.read()
        return f"data:image/png;base64,{base64.b64encode(file).decode()}"


@st.cache_data
def get_df_with_images(initial_df):
    df = initial_df.copy()
    image_getter = partial(get_image_path, folder = 'Dataset/images/image_train')
    image_paths = df.apply(image_getter, axis=1)
    df.insert(0, 'image', image_paths)
    df["image"] = df['image'].apply(image_path_to_base64)
    return df


def reset_sample():
    del st.session_state['sample']


def show_demo(sample_size = 120,  image_size = 100):
    st.header("🚀 Démo interactive")
    X, y = load_dataset()

    # Pick a sample from X (because images would use too many resources for the whole X)
    if 'sample' not in st.session_state:
        sample = X.sample(sample_size)
        # st.session_state['sample'] = get_df_with_images(sample)
        st.session_state['sample'] = sample

    # Product selection

    # st.markdown(f"## Choix du produit")
    st.markdown(f"Veuillez cocher un produit à catégoriser par le modèle.\nPour consulter les détails des produits, vous pouvez faire défiler le tableau horizontalement/verticalement, ou le mettre en plein écran.")

    st.button("Regénérer les produits", on_click=reset_sample)


    event = st.dataframe(st.session_state['sample'],
                #  column_config={'image': st.column_config.ImageColumn(width= image_size)},
                #  row_height= image_size,
                #  height=400,  # when specified, hinders full screen mode
                 on_select="rerun",
                 selection_mode="single-row")

    # If user has selected a product
    if event.selection.rows:
        # Get index of user-selected row
        input_index = event.selection.rows[0]

        # Get user-selected row at proper index (convert index from `session_state['sample'].iloc` to `X.loc`)
        row_index = int(st.session_state['sample'].iloc[input_index].name)
        X_row = X.loc[[row_index]]  # Double-bracket: to keep it as a dataframe instead of a series
        y_row = y.loc[[row_index]]  # Double-bracket: to keep it as a series

        # Prediction
        y_pred_class = predict(X_row)
        y_class = y_row.iloc[0]
        y_pred_description = get_class_description(y_pred_class)
        y_description = get_class_description(y_class)

        # Display prediction
        if y_pred_class == y_class:
            prediction_style = "green"
        else:
            prediction_style = "red"

        _, col1, col2, _ = st.columns(4)
        with col1:
            st.markdown(f"### :green[catégorie]\n:green[{y_class} - {y_description}]")
        with col2:
            st.markdown(f"### :{prediction_style}[prédiction]\n:{prediction_style}[{y_pred_class} - {y_pred_description}]")


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_demo()
