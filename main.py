from loadData import DataLoader
from dataPreprocessor import DataPreprocessor
import os
import numpy as np

def main1():
    #!!!! PONGO MI MAIN QUE SE QUE FUNCIONA CON EL LOAD.
    print("--- CARREGAR DADES ---")
    loader = DataLoader()
    dataframe3 = loader.load_csv("data/features_3_sec.csv")
    dataframe30 = loader.load_csv("data/features_30_sec.csv")
    # print(dataframe.head())

    DIR_IMAGES = "data/images_original"
    genre_imgs = {} #??? hago un diccionario

    for dir_genre in os.listdir(DIR_IMAGES):
        genre_path = os.path.join(DIR_IMAGES, dir_genre)
        genre_imgs[dir_genre] = loader.load_img(genre_path)




def main2():

    # Paths dels datasets
    #* poso els paths relatius pero si no funciona correctament podem posar el path complet URL (el del git)
    path_csv_3s = r"data/features_3_sec.csv" 
    path_csv_30s = r"data/features_30_sec.csv"
    images_path = r"data/images_original"
    audios_path = r"data/genres_original"

    # Definir columna que s'utilitzara com a variable de sortida (Y)--> conté els generes musicals
    # La resta de columnes son les caracteristiques que el model utilitza com entrada (X)
    TARGET_COLUMN = "label" 

    print("--- CARREGAR DADES ---")
    loader = DataLoader()
    df_3s = loader.load_csv(path_csv_3s)
    df_30s = loader.load_csv(path_csv_30s)
    #X_images, genres = loader.load_img(images_path)
    genre_imgs = {}
    for dir_genre in os.listdir(images_path):
        genre_path = os.path.join(images_path, dir_genre)
        genre_imgs[dir_genre] = loader.load_img(genre_path)

    genres = list(genre_imgs.keys())
    X_images = np.array([img for imgs in genre_imgs.values() for img in imgs.values()])

    data = DataPreprocessor()
    X_3s, y_3s, classes_3s = data.preprocess_csv(df_3s, TARGET_COLUMN)
    X_3s_normalized = data.normalize_data(X_3s)
    X_3s_denoised = data.remove_noise(X_3s_normalized)
    X_3s_train, X_3s_test, y_3s_train, y_3s_test = data.split_data(X_3s_denoised, y_3s)
    #print(f"CSV 3s carregat. Shape X: {X_3s.shape}, Num etiquetes: {len(classes_3s)}")
    print(f"CSV 3s carregat. Train shape: {X_3s_train.shape}, Test shape: {X_3s_test.shape}")

    X_30s, y_30s, classes_30s = data.preprocess_csv(df_30s, TARGET_COLUMN)
    X_30s_normalized = data.normalize_data(X_30s)
    X_30s_denoised = data.remove_noise(X_30s_normalized)
    X_30s_train, X_30s_test, y_30s_train, y_30s_test = data.split_data(X_30s_denoised, y_30s)
    #print(f"CSV 30s carregat. Shape X: {X_30s.shape}, Num etiquetes: {len(classes_30s)}")
    print(f"CSV 30s carregat. Train shape: {X_30s_train.shape}, Test shape: {X_30s_test.shape}")

    y_images, classes_images = data.preprocess_images(genres)
    print(f"Imatges carregades. Shape X: {X_images.shape}, Num etiquetes: {len(classes_images)}")
    

if __name__ == "__main__":
    #main1() #! para que asi cree los pickle de las imagenes.
    main2()