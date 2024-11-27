"""
FUNCIÓ PRINCIPAL:
Aquest script serveix per obtenir dades tant de les imatges, csv o audios.

"""

import pandas as pd
import os 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler
import cv2


# Paths dels datasets
#* poso els paths relatius pero si no funciona correctament podem posar el path complet URL (el del git)
path_csv_3s = r"data\features_3_sec.csv" 
path_csv_30s = r"data\features_30_sec.csv"
images_path = r"data\images_original"
audios_path = r"data\genres_original"

# Definir columna que s'utilitzara com a variable de sortida (Y)--> conté els generes musicals
# La resta de columnes son les caracteristiques que el model utilitza com entrada (X)
TARGET_COLUMN = "label" 

def load_csv_data(csv_path, target_column):
    """
    Carrega les dades del csv, separa les característiques (X) i les etiquetes (y).
    Returns: X, y i els noms de les característiques. 
    """
    # Carregar csv en un dataframe
    data = pd.read_csv(csv_path)

    # Separem caracteristiques i etiquetes
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Codificar etiquetes (string --> num | blues --> 0)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder.classes_

X_3s, y_3s, classes_3s = load_csv_data(path_csv_3s, TARGET_COLUMN)
X_30s, y_30s, classes_30s = load_csv_data(path_csv_30s, TARGET_COLUMN)

print(f"Dades 3s: {X_3s.shape}, Etiquetes: {len(np.unique(y_3s))}")


def load_images(image_dir, image_size=(128,128)):
    """
    Carrega imatges des d'un directori i les transforma en matrius.

    image_size: mida de les imatges perque totes siguin uniformes abans de procesarles
    """
    images = []
    genres = []

    for genre_folder in os.listdir(image_dir):
        genre_path = os.path.join(image_dir, genre_folder)
        if os.path.isdir(genre_path):
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)

                # Llegir la imatge
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)

                    # Convertim a RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    images.append(img/255.0) # Normalitzar 
                    genres.append(genre_folder)

    # Convertir a Numpy
    images = np.array(images, dtype='float32') 
    genres = np.array(genres)

    # Codificar etiquetes
    encoder = LabelEncoder()
    genres_encoded = encoder.fit_transform(genres)

    return images, genres_encoded, encoder.classes_

X_images, y_images, classes_images = load_images(images_path)
print(f"Imatges carregades: {X_images.shape}, Etiquetes: {len(np.unique(y_images))}")
print(y_images)
