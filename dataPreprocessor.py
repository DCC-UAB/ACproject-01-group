"""
FUNCIÓ PRINCIPAL:
Aquest script serveix per obtenir dades tant de les imatges, csv o audios.
A més a més preprocessarem les dades del csv, normalitzant-les, eliminant el soroll i dividint en train i test.

"""

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class DataPreprocessor:
    def preprocess_csv(self, df, target_column):
        """
        Separem característiques (X) i etiquetes (Y) d'un DataFrame i codifiquem les etiquetes.
        """

        # Separem caracteristiques i etiquetes
        X = df.drop(columns=[target_column])
        y = df[target_column]

    
        # Codificar etiquetes (string --> num | blues --> 0)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        return X, y_encoded, encoder.classes_
        

    def preprocess_images(self, genres):
        """
        Codifiquem les etiquetes de les imatges.
        """

        # Codificar etiquetes
        encoder = LabelEncoder()
        genres_encoded = encoder.fit_transform(genres)

        return genres_encoded, encoder.classes_

    def preprocess_audio(self):
        pass

    def normalize_data(self, X):
        """
        Normalitzem les dades utilitzant MinMaxScaler.
        """
        scaler = MinMaxScaler()
        X_numeric = X.select_dtypes(include=[np.number])  # Selecciona només columnes numèriques
        X_normalized = scaler.fit_transform(X_numeric)
        X_normalized_df = pd.DataFrame(X_normalized, columns=X_numeric.columns)
        return X_normalized_df
    
    def remove_noise(self, X, treshold=1e-6):
        """
        Eliminem el soroll (valors molt petits) de les dades establint a 0 aquells valors que siguin menor que el llindar donat.
        """
        X_cleaned = np.where(np.abs(X) < treshold, 0, X)
        return X_cleaned
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Divideix les dades en conjunt d'entrenament i test.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    