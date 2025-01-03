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
import os, pickle
import pandas as pd
from PIL import Image #? no se si funcionará bien como librosa (la hemos utilizado en otra asignatura: PSIV)
import numpy as np

class DataLoader:
    def __init__(self, cache_dir="cache_data"):
        self.cache_dir = cache_dir
        # Crear la carpeta si no existeix
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_csv(self, csv_path:str) -> pd.DataFrame:
        """
        Carreguem l'arxiu i l'emmagatzema si no existeix.
        Retorna el pandas. 
        """
        filename_cache = os.path.basename(csv_path).replace(".csv", ".pkl") # nomFitxer.pkl
        cache_path = os.path.join(self.cache_dir, filename_cache) # El path: /cache_data/nomFitxer.pkl

        #* Si el fitxer pickle ja existeix, el carreguem des del cache
        if os.path.exists(cache_path):
            print(f"Carregant dades des de CACHÉ: {cache_path}")
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
    
    def load_wav(self):
        pass

    def load_img(self, path_imgs:str) -> dict:
        """
        Carreguem les imatges d'una carpeta (gènere)
        Retorna un dict amb el format {nom_arxiu : numpy}
        """
        dir_name = os.path.basename(os.path.normpath(path_imgs))  # nom de la carpeta, p.e: 'pop'
        cache_path = os.path.join(self.cache_dir, f"{dir_name}_images.pkl") # p.e 'cache_data/pop_images.pkl'

        #* Intentar carregar des de cache
        if os.path.exists(cache_path):
            print(f"Carregant imatges des de CACHÉ: {cache_path}")
            with open(cache_path, "rb") as f:
                images = pickle.load(f)
        else:
            print(f"Carregant imatges des de la carpeta: {path_imgs}")
            images = {}

            for imgname in os.listdir(path_imgs):
                file_path = os.path.join(path_imgs, imgname) # el path complet /data/.../imgname
                try:
                    img = Image.open(file_path)
                    images[imgname] = np.array(img)
                except Exception as e:
                    print(f"Error carregant la imatge{imgname}: {e}")

            with open(cache_path, "wb") as f:
                pickle.dump(images, f)
            print(f"{len(images)} imatges carregades desde {path_imgs}.")
        
        return images