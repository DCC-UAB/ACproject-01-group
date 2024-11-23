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
import pandas as pd
import pickle

class DataLoader:
    def __init__(self, cache_dir="cache_data"):
        self.cache_dir = cache_dir

    def load_csv(self, csv_path:str) -> pd.DataFrame:
        """
        Carreguem l'arxiu ARXIUS i l'emmagatzema si no existeix.
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
    
    def load_wav():
        pass

    def load_img():
        pass
