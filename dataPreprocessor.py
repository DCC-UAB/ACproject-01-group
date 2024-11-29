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
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None

    def preprocess_csv(self, df, target_column):
        """
        Separem característiques (X) i etiquetes (Y) d'un DataFrame i codifiquem les etiquetes.
        """

        # Separem caracteristiques i etiquetes
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]

    
        # Codificar etiquetes (string --> num | blues --> 0)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(self.y)

        return y_encoded, encoder.classes_
        

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

    def normalize_data(self):
        """
        Normalitzem les dades utilitzant MinMaxScaler.
        """
        scaler = MinMaxScaler()
        X_numeric = self.X.select_dtypes(include=[np.number])  # Selecciona només columnes numèriques
        X_normalized = scaler.fit_transform(X_numeric)
        X_normalized_df = pd.DataFrame(X_normalized, columns=X_numeric.columns)
        return X_normalized_df
    
    def remove_noise(self, treshold=1e-6):
        """
        Eliminem el soroll (valors molt petits) de les dades establint a 0 aquells valors que siguin menor que el llindar donat.
        """
        self.X = np.where(np.abs(X) < treshold, 0, self.X) # la X_cleaned = a la nostra X definitiva (posem self.x)
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Divideix les dades en conjunt d'entrenament i test.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
    
    