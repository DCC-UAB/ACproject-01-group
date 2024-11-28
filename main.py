from loadData import DataLoader
from dataPreprocessor import DataPreprocessor
import os
import numpy as np

def main():
    PATH_CSV3 = r"data/features_3_sec.csv" 
    PATH_CSV30 = r"data/features_30_sec.csv"
    PATH_IMAGES = r"data/images_original"
    PATH_AUDIOS = r"data/genres_original"

    # Definir columna que s'utilitzara com a variable de sortida (Y)--> conté els generes musicals
    TARGET_COLUMN = "label" 

    print("--- CARREGAR DADES ---")
    loader = DataLoader()
    df_3s = loader.load_csv(PATH_CSV3)
    df_30s = loader.load_csv(PATH_CSV30)
    # print(dataframe30.head())

    genre_imgs = {}
    for dir_genre in os.listdir(PATH_IMAGES):
        genre_path = os.path.join(PATH_IMAGES, dir_genre)
        genre_imgs[dir_genre] = loader.load_img(genre_path)

    genres = list(genre_imgs.keys())
    X_images = np.array([img for imgs in genre_imgs.values() for img in imgs.values()])

    print("\n--- PRE-PROCESSAR ---")
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
    main()