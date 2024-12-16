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
import matplotlib.pyplot as plt
import os


class DataPreprocessor:
    def __init__(self):
        self._X = None
        self._y = None

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

        self._normalized_x_train = None
        self._normalized_x_test = None

        self._encoder = LabelEncoder()
        self._y_encoded = None

    @property
    def train_data(self):
        if self._normalized_x_train is not None:
            return self._normalized_x_train
        elif self._X_train is not None:
            return self._X_train
        else:
            raise ValueError(
                "Les dades train encara no han estat generades. Crida a 'split_data()'"
            )

    # Setter per poder assignar nous valors a train_data
    @train_data.setter
    def train_data(self, value):
        self._X_train = value

    @property
    def test_data(self):
        if self._normalized_x_test is not None:
            return self._normalized_x_test
        elif self._X_test is not None:
            return self._X_test
        else:
            raise ValueError(
                "Les dades test encara no han estat generades. Crida a 'split_data()'"
            )

    # Setter per poder assignar nous valors a test_data
    @test_data.setter
    def test_data(self, value):
        self._X_test = value

    @property
    def test_labels(self):
        if self._y_test is None:
            raise ValueError(
                "Les dades test encara no han estat generades. Crida a 'split_data()'"
            )
        return self._y_test

    @property
    def train_labels(self):
        if self._y_train is None:
            raise ValueError(
                "Les dades train encara no han estat generades. Crida a 'split_data()'"
            )
        return self._y_train

    def get_labels(self) -> list:
        if self._y is None:
            raise ValueError(
                "Encara no s'ha preprocessat. Crida a 'preprocess_csv(df)'"
            )
        return list(self._encoder.classes_)

    def preprocess_csv(self, df: pd.DataFrame) -> None:
        """
        Separem característiques (X) i etiquetes (Y) d'un DataFrame i codifiquem les etiquetes.
        """
        # Separem
        self._X = df.iloc[:, 1:-1]  # descartem les columnes filename i target
        self._y = df.iloc[:, -1]  # seleccionem unicament la columna target

        # Codificar (string --> num | blues --> 0)
        self._y_encoded = self._encoder.fit_transform(self._y)

    def preprocess_images(self, images: dict, target_size=(64, 64)):
        """
        Codifiquem les etiquetes de les imatges i obtenim els arrays numpys.

        images: diccionari {nom_arxiu: numpy array de les imatges}
        target_size: mida a la que volem redimensionar les imatges

        """
        X = []  # guarda les imatges redimesionades
        y = []  # guarda les etiquetes

        for genre, image_dict in images.items():
            # Iterem sobre cada arxiu de imatge dins d'un genere
            for filename, img_array in image_dict.items():
                # Redimensionem la imatge
                img = Image.fromarray(img_array)

                # Redimensionar la imatge
                img_resized = img.resize(target_size)
                img_resized_array = np.array(img_resized)  # Convertir a numpy

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

    def split_data(self, test_size=0.2, random_state=42):
        """
        Divideix les dades en conjunt d'entrenament i test.
        """
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X, self._y, test_size=test_size, random_state=random_state
        )

        # Convertir a numpy
        self._X_train = np.asarray(self._X_train)
        self._y_train = np.asarray(self._y_train)
        self._X_test = np.asarray(self._X_test)
        self._y_test = np.asarray(self._y_test)

    def normalize_data(self):
        """
        Normalitzem les dades utilitzant MinMaxScaler / Normalizer (no va bien para este conjunto) / StandardScaler / RobustScaling.
        """
        # scaler = StandardScaler().fit(self._X_train)
        scaler = MinMaxScaler().fit(self._X_train)

        self._normalized_x_train = scaler.transform(self._X_train)
        self._normalized_x_test = scaler.transform(self._X_test)

    def remove_noise(self, treshold=1e-6):
        """
        Eliminem el soroll (valors molt petits) de les dades establint a 0 aquells valors que siguin menor que el llindar donat.
        """
        self._X = np.where(
            np.abs(self._X) < treshold, 0, self._X
        )  # la X_cleaned = a la nostra X definitiva (posem self.x)

    def plot_features(self, filename: str):
        """
        Genera una imatge png on es veuen totes les caracteristiques (features) del dataset contra les etiquetes.
        """
        #! evitem sobreescriure si ja existeix
        if os.path.exists(filename):
            print(
                f"L'arxiu {filename} ja existeix existe, per tant, no es genera una imatge nova."
            )
            return

        num_features = self._X.shape[1]  # columnes

        # Crear un grafic amb subgrafics (un per cada feature) de 3 "columnes"
        fig, axes = plt.subplots(
            nrows=(num_features // 3) + 1,
            ncols=3,
            figsize=(15, 5 * (num_features // 3 + 1)),
        )
        axes = axes.flatten()

        for i, column in enumerate(self._X.columns):
            ax = axes[i]
            ax.scatter(self._X[column], self._y)
            ax.set_xlabel(column)
            ax.set_ylabel("Label")
            ax.set_title(column)

        # * Eliminar els subgrafics buits
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Ajustar
        plt.tight_layout()

        # Guardar la imagen
        plt.savefig(filename, format="png")
        print(f"Imatge guardada com a {filename}")
