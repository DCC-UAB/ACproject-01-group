"""
FUNCIÓ PRINCIPAL:
Aquest script serveix per obtenir dades tant de les imatges, csv o audios.

"""

import pandas as pd
import os 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    Retorna X, y i els noms de les característiques. 
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