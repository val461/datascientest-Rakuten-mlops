"""
Script pour générer automatiquement une nouvelle page Streamlit dans le dossier `parts` et mettre à jour le fichier `app.py`.
Usage: python generate_page.py

Après lancement de ce script, merci de vérifier les changements générés dans le fichier app.py avant de les commit !
"""

import os
import re


def slugify(text):
    """Convertit un titre en nom de fichier/fonction valide"""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text).strip().lower()
    return text


def update_app_py(titre_complet, titre_abrege, nom_fonction):
    """Met à jour automatiquement app.py"""
    app_path = "app.py"

    if not os.path.exists(app_path):
        print("⚠️  app.py n'existe pas, création manuelle nécessaire")
        return

    with open(app_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Trouver où insérer l'import
    import_line = f"from parts.{titre_abrege} import {nom_fonction}\n"
    pages_line = f'    "{titre_complet}": {nom_fonction},\n'

    new_lines = []
    import_added = False
    pages_added = False

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Ajouter l'import après les autres imports
        if not import_added and line.startswith("from parts.") and i + 1 < len(lines):
            if not lines[i + 1].startswith("from parts."):
                new_lines.append(import_line)
                import_added = True

        # Ajouter dans PAGES
        if not pages_added and '"' in line and '":' in line and 'PAGES' in ''.join(lines[0:i]):
            # On est dans le dict PAGES, ajouter avant le dernier }
            if i + 1 < len(lines) and '}' in lines[i + 1]:
                print('debug 2')
                new_lines.append(pages_line)
                pages_added = True

    if import_added or pages_added:
        with open(app_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("✅ app.py mis à jour automatiquement !")
        if not import_added:
            print(f"⚠️  Échec de l'ajout de la ligne d'import, faites-le manuellement :\n{import_line}")
        if not pages_added:
            print(f"⚠️  Échec de l'ajout de la ligne de page, faites-le manuellement :\n{pages_line}")
    else:
        print("⚠️  Mise à jour automatique impossible, faites-le manuellement : {import_line}{pages_line}")


def generate_page():
    print("=== Générateur de page Streamlit ===\n")

    titre_complet = input("Titre complet (ex: 🎯 Problématique): ").strip()
    titre_abrege = input(f"Titre python (ex: problematique): ").strip()

    nom_fichier = f"{titre_abrege}.py"
    nom_fonction = f"show_{titre_abrege}"
    chemin_fichier = os.path.join("parts", nom_fichier)

    # Vérifier si existe déjà
    if os.path.exists(chemin_fichier):
        overwrite = input(f"⚠️  {chemin_fichier} existe déjà. Écraser ? (o/N): ")
        if overwrite.lower() != 'o':
            print("❌ Annulé")
            return

    contenu = f'''# parts/{nom_fichier}

import streamlit as st


def {nom_fonction}():
    st.title("{titre_complet}")
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
    {nom_fonction}()
'''

    os.makedirs("parts", exist_ok=True)

    with open(chemin_fichier, 'w', encoding='utf-8') as f:
        f.write(contenu)

    print(f"\n✅ Fichier créé : {chemin_fichier}\n")
    print("📝 À ajouter dans app.py :")
    print(f"from parts.{titre_abrege} import {nom_fonction}")
    print(f'    "{titre_complet}": {nom_fonction},')
    print()

    auto_update = input("Mettre à jour app.py automatiquement ? (O/n): ")
    if auto_update.lower() != 'n':
        update_app_py(titre_complet, titre_abrege, nom_fonction)


if __name__ == "__main__":
    generate_page()
