# Pour lancer cette application :
# se placer dans le dossier du repo git, puis lancer les commandes suivantes.
# source venv/bin/activate
# streamlit run src/streamlit.py

import sys
from pathlib import Path

# Quick fix for import issues
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from tensorflow import keras
import streamlit as st
import base64
from pathlib import Path
from functools import partial
from src.preprocessing.image import get_image_path
from src.models.on_text_and_images.deep_learning import CATEGORY_MAPPING, grad_cam
from src.preprocessing.core import load_reproducible_split
from src.preprocessing.pipelines.deep_learning import load_preprocessors
from src.preprocessing.pipelines.deep_learning_on_text_and_images import preprocess_features


@st.cache_data
def load_Dataset2():
    X_train, X_test, y_train, y_test = load_reproducible_split(folder = 'Dataset2')
    return X_train, X_test, y_train, y_test


@st.cache_resource
def preprocessing_DL3(X_test, y_test):
    # parameters
    multimodal_artifacts_folder = Path(f'artifacts/on_text_and_images/deep_learning/v1')
    preprocessors_folder = multimodal_artifacts_folder
    BATCH_SIZE = 32

    preprocessors = load_preprocessors(names=['target','tabular','hash','text_vectorizer'], artifacts_folder=preprocessors_folder)

    test_ds, new_preprocessors, class_weights_test, test_inputs_dict, y_test_ohe = preprocess_features(X_test, y_test, preprocessors, shuffle=False, BATCH_SIZE = BATCH_SIZE, rebalance_with_weights=False, augment=False)

    if new_preprocessors:
        print(f"error: some preprocessors got fitted on the testing set, so they were probably not handled properly. {new_preprocessors=}")

    return test_ds, preprocessors


@st.cache_resource
def load_model_DL3():
    path = "artifacts/on_text_and_images/deep_learning/v1/best_model_arch-11_epoch_index-01_val_accuracy-0.8338_f1-0.8337.keras"
    model = keras.models.load_model(path)
    return model


def np_array_to_int(np_array):
    return int(np_array[0])


def predict_DL3(model, test_ds, preprocessors):
    y_pred = model.predict(test_ds)
    y_pred_class = np_array_to_int(preprocessors['target'].inverse_transform(y_pred.argmax(axis=1)))
    return y_pred_class


@st.cache_data
def get_class_description(class_code: int):
    description = CATEGORY_MAPPING.get(class_code, "inconnue")
    return description


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


def get_grad_cam_images(inputs_batch_dict, model, conv_layers):
    """
    Retourne l'image originale et les images Grad-CAM.

    Args:
        inputs_batch_dict: Le batch d'entrées (dictionnaire).
        model: Le modèle Keras.
        conv_layers: Liste des noms de couches à visualiser.
        true_labels: Les vrais labels (One-Hot encoded).
        hash_encoder: L'OrdinalEncoder pour décoder les MD5.
        target_encoder: Le LabelEncoder pour décoder les classes (Int -> String).
        category_mapping: dict describing classes.
    """
    original_image = inputs_batch_dict['image_input'][0].numpy().astype("uint8")
    grad_cam_images = {}
    grad_cam_images['original'] = original_image

    for layer_name in conv_layers:
        # On isole un échantillon unique pour grad_cam
        single_sample = {key: value[0] for key, value in inputs_batch_dict.items()}
        try:
            grad_cam_image, predicted_index = grad_cam(single_sample, model, layer_name)
            grad_cam_images[layer_name] = grad_cam_image
        except Exception as e:
            st.text(f"Erreur sur {layer_name}:\n{e}")

    return grad_cam_images


def reset_sample():
    del st.session_state['sample']


def show_demo(sample_size = 15, small_image_size = 100):
    st.title("Catégorisation de produits Rakuten par deep learning multi-modal")
    st.header("🚀 Démo interactive")
    X_train, X_test, y_train, y_test = load_Dataset2()

    # Pick a sample from X_test (because images would use too many resources for the whole X_test)
    if 'sample' not in st.session_state:
        sample = X_test.sample(sample_size)
        st.session_state['sample'] = get_df_with_images(sample)

    # Product selection
    # st.markdown(f"## Choix du produit")
    st.markdown(f"Veuillez cocher un produit à catégoriser par le modèle DL3.\nPour consulter les détails des produits, vous pouvez faire défiler le tableau horizontalement/verticalement, ou le mettre en plein écran.")

    # Allow refreshing sample
    st.button("Regénérer les produits", on_click=reset_sample)

    event = st.dataframe(st.session_state['sample'],
                 column_config={'image': st.column_config.ImageColumn(width=small_image_size)},
                 row_height=small_image_size,
                #  height=400,  # when specified, hinders full screen mode
                 on_select="rerun",
                 selection_mode="single-row")

    # If user has selected a product
    if event.selection.rows:
        # Get index of user-selected row
        input_index = event.selection.rows[0]

        # Get user-selected row at proper index (convert index from `session_state['sample'].iloc` to `X_test.loc`)
        row_index = int(st.session_state['sample'].iloc[input_index].name)
        X_test_row = X_test.loc[[row_index]]  # Double-bracket: to keep it as a dataframe instead of a series
        y_test_row = y_test.loc[[row_index]]  # Double-bracket: to keep it as a series

        # st.write("Produit sélectionné :", st.session_state['sample'].iloc[input_index])
        # st.write(row_index,X_test_row,f"{y_test_row.iloc[0]=} {type(y_test_row)=}")

        # Prediction
        test_ds, preprocessors = preprocessing_DL3(X_test_row, y_test_row)
        model = load_model_DL3()
        y_pred_class = predict_DL3(model, test_ds, preprocessors)
        y_test_class = y_test_row.iloc[0]
        y_pred_description = get_class_description(y_pred_class)
        y_test_description = get_class_description(y_test_class)

        # Display prediction
        # st.markdown(f"## Prédiction")
        if y_pred_class == y_test_class:
            prediction_style = "green"
        else:
            prediction_style = "red"

        _, col1, col2, _ = st.columns(4)
        with col1:
            st.markdown(f"### :green[catégorie]\n:green[{y_test_class} - {y_test_description}]")
        with col2:
            st.markdown(f"### :{prediction_style}[prédiction]\n:{prediction_style}[{y_pred_class} - {y_pred_description}]")

        # Grad-CAM

        st.markdown(f"## Interprétation Grad-CAM")
        for inputs_dict, labels_batch in test_ds: # type: ignore
            labels = labels_batch.numpy() #.argmax(axis=1)
            break

        # st.write(inputs_dict.keys(), labels.shape) # type: ignore
        selected_layers=['block3b_project_conv', 'block5e_project_conv', 'top_conv']  # manual selection of layers
        grad_cam_images = get_grad_cam_images(inputs_dict, model, selected_layers)  # type: ignore
        cols = st.columns(len(grad_cam_images))
        for k, (layer, grad_cam_image) in enumerate(grad_cam_images.items()):
            with cols[k]:
                # st.markdown(f"<div style='text-align: center'>{selected_layers[k]}</div>", unsafe_allow_html=True)
                st.image(grad_cam_image, caption=f"{layer}")


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(
        page_title="Projet Rakuten - catégorisation multi-modale",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    show_demo()
