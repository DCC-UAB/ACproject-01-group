"""
FUNCIÓ PRINCIPAL:
Aquest script serveix per obtenir dades tant de les imatges, csv o audios.
A més a més preprocessarem les dades del csv, normalitzant-les, eliminant el soroll i dividint en train i test.

"""
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image

class DataPreprocessor:
    def __init__(self):
        self._X = None
        self._y = None
        self._X_train = None
        self._X_test = None
        self._y_train = None 
        self._y_test = None

        self._encoder = LabelEncoder()

    @property
    def train_data(self):
        if self._X_train is None:
            raise ValueError("Les dades train encara no han estat generades. Crida a 'split_data()'")
        return self._X_train
    @property
    def test_data(self):
        if self._X_test is None:
            raise ValueError("Les dades test encara no han estat generades. Crida a 'split_data()'")
        return self._X_test
    @property
    def test_labels(self):
        if self._y_test is None:
            raise ValueError("Les dades test encara no han estat generades. Crida a 'split_data()'")
        return self._y_test
    @property
    def train_labels(self):
        if self._y_train is None:
            raise ValueError("Les dades train encara no han estat generades. Crida a 'split_data()'")
        return self._y_train
    
    
    def preprocess_csv(self, df:pd.DataFrame, target_column:str):
        """
        Separem característiques (X) i etiquetes (Y) d'un DataFrame i codifiquem les etiquetes.
        """
        # Separem
        self._X = df.drop(columns=[target_column])
        self._y = df[target_column]

        # Codificar (string --> num | blues --> 0)
        y_encoded = self._encoder.fit_transform(self._y)

        return y_encoded, self._encoder.classes_
        

    def preprocess_images(self, images:dict, target_size:tuple):
        """
        Codifiquem les etiquetes de les imatges i obtenim els arrays numpys.

        images: diccionari {nom_arxiu: numpy array de les imatges}
        target_size: mida a la que volem redimensionar les imatges 

        """
        X = [] # guarda les imatges redimesionades
        y = [] # guarda les etiquetes
        for genre, image_dict in images.items():
            # Iterem sobre cada arxiu de imatge dins d'un genere
            for filename, img_array in image_dict.items():
                # Redimensionem la imatge 
                img = Image.fromarray(img_array)

                # Redimensionar la imatge
                img_resized = img.resize(target_size)
                img_resized_array = np.array(img_resized) # Convertir a numpy

                # Normalitzem la imatge
                img_resized_array = img_resized_array / 255.0 

                X.append(img_resized_array)
                y.append(genre)
        
        self._X = np.array(X)
        self._y = np.array(y)

        # Codificar etiquetes
        self._y = self._encoder.fit_transform(self._y)

    def preprocess_audio(self):
        pass

    def normalize_data(self) -> None:
        """
        Normalitzem les dades utilitzant MinMaxScaler.
        """
        scaler = MinMaxScaler()

        X_numeric = self._X.select_dtypes(include=[np.number])  # Selecciona només columnes numèriques
        self._X = scaler.fit_transform(X_numeric) #! actualitzem
    
    def remove_noise(self, treshold=1e-6):
        """
        Eliminem el soroll (valors molt petits) de les dades establint a 0 aquells valors que siguin menor que el llindar donat.
        """
        self._X = np.where(np.abs(self._X) < treshold, 0, self._X) # la X_cleaned = a la nostra X definitiva (posem self.x)
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Divideix les dades en conjunt d'entrenament i test.
        """
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y, test_size=test_size, random_state=random_state)
    