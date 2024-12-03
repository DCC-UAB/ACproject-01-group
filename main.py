from loadData import DataLoader
from dataPreprocessor import DataPreprocessor
from algorithms import Models
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

    print("\n--- PRE-PROCESSAR ---")
    data3 = DataPreprocessor()
    data3.preprocess_csv(df_3s, TARGET_COLUMN) #??? valors encoded - no seria mejor de cada conjunto train i test
    data3.normalize_data()
    data3.remove_noise() #!!!! provar amb diferents thresholds, per decidir quin es el millor
    data3.split_data()
    print(f"CSV 3s carregat. Train shape: {data3.train_data.shape}, Test shape: {data3.test_data.shape}")

    data30 = DataPreprocessor()
    data30.preprocess_csv(df_30s, TARGET_COLUMN)
    data30.normalize_data()
    data30.remove_noise()
    data30.split_data()
    print(f"CSV 30s carregat. Train shape: {data30.train_data.shape}, Test shape: {data30.test_data.shape}")

    #? creo que no esta bien hecho el preprocess para img
    # genres = list(genre_imgs.keys())
    # X_images = np.array([img for imgs in genre_imgs.values() for img in imgs.values()])
    # dataIMG = DataPreprocessor
    # y_images, classes_images = dataIMG.preprocess_images(genres)
    # print(f"Imatges carregades. Shape X: {X_images.shape}, Num etiquetes: {len(classes_images)}")

    print("\n--- IMPLEMENTAR MODELS ---")
    models3 = Models(data3.train_data, data3.train_labels, data3.test_data, data3.test_labels)

    #* Entrenar els models
    models3.do_decision_tree()
    models3.do_random_forest()
    models3.do_gradient_boosting()

    #* Avaluar els models
    models3.evaluate_model('Decision Tree', dataset_name="df_3s")
    models3.evaluate_model('Random Forest', dataset_name="df_3s")
    models3.evaluate_model('Gradient Boosting', dataset_name="df_3s")

    metrics_df = models3.create_metrics_dataframe()
    

if __name__ == "__main__":
    main()