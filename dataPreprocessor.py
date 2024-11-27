"""
FUNCIÓ PRINCIPAL:
Aquest script serveix per obtenir dades tant de les imatges, csv o audios.

"""

from sklearn.preprocessing import LabelEncoder
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
