"""
FUNCIÓ PRINCIPAL:
Aquest script serveix per carregar les dades de la carpeta 'data' en un arxiu pickle i preprocessar les dades.

INFORMACIÓ D'INTERÈS:
L'arxiu pickle basicament converteix un objecte de Python en un format binari que es pot guardar en un arxiu o transmetre. 
És a dir, que pot tant emmagatzemar en binari com recuperar la seva forma original.

S'ha decidit utilitzar-ho, degut a que a l'estar tractant amb grans fitxers i varis models, creiem que és una bona idea 
guardar-ho, per després carregar-ho sense necessitat de processar les dades des de cero repetitivament. D'aquesta manera,
optimitzarem (serà més ràpid) el temps de processament 

"""

#???????????????? TAMBIEN PODRIAMOS USAR __PYCACHE__ -> EVITAMOS RECOMPILAR CODIGO QUE SE MATIENE
import os
import pickle
import pandas as pd
import cv2
import numpy as np

class DataLoader:
    def __init__(self, cache_dir="cache_data", image_size=(128,128)):
        """
        cache_dir: directori on es guarden les dades
        image_size: mida de les imatges perque totes siguin uniformes abans de procesarles
        """
        self.cache_dir = cache_dir
        self.image_size = image_size

    def load_csv(self, csv_path:str) -> pd.DataFrame:
        """
        Carreguem arxiu csv i l'emmagatzema si no existeix 
        """
        filename_cache = os.path.basename(csv_path).replace(".csv", ".pkl") # nomFitxer.pkl
        cache_path = os.path.join(self.cache_dir, filename_cache) # El path: /cache_data/nomFitxer.pkl

        #* Si el fitxer pickle ja existeix, el carreguem des del cache
        if os.path.exists(cache_path):
            print(f"Carregant dades des de caché: {cache_path}")
            with open(cache_path, "rb") as f: #! rb = read binary
                df = pickle.load(f)
        
        else:
            print(f"Carregant dades des de CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            #* L'escrivim
            with open(cache_path, "wb") as f: #! wb = write binary
                pickle.dump(df, f)
            print(f"Dades cacheades en: {cache_path}")

        return df
        
    
    def load_images(self, image_dir):
        """
        Carrega imatges des d'un directori i les transforma en matrius.
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
                        img = cv2.resize(img, self.image_size)

                        # Convertim a RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        images.append(img/255.0) # Normalitzar 
                        genres.append(genre_folder)

        # Convertir a Numpy
        images = np.array(images, dtype='float32') 
        genres = np.array(genres)

        
        return images, genres

    def load_audios():
        pass
