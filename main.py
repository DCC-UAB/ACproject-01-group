from loadData import DataLoader
from dataPreprocessor import DataPreprocessor
from algorithms import Models
import os
import pandas as pd
import pickle

def get_models(model:object) -> dict:
        return {
        'KNN': model.do_knn,
        'SVM': model.do_svm,
        'Decision Tree': model.do_decision_tree,
        'Random Forest': model.do_random_forest,
        'Gradient Boosting': model.do_gradient_boosting,
        'Logistic Regression': model.do_logistic_regression,
        'Gaussian NB': model.do_gaussian_naive_bayes,
        'Bernoulli NB': model.do_bernoulli_naive_bayes,
        'Multinomial NB': model.do_multinomial_nb,
        }

def main():
    PATH_CSV3 = r"data/features_3_sec.csv" 
    PATH_CSV30 = r"data/features_30_sec.csv"
    PATH_IMAGES = r"data/images_original"
    PATH_AUDIOS = r"data/genres_original"

    print("--- CARREGAR DADES ---")
    loader = DataLoader()
    df_3s = loader.load_csv(PATH_CSV3)
    df_30s = loader.load_csv(PATH_CSV30)

    genre_imgs = {}
    # Carregar les imatges dels directoris per genere
    for dir_genre in os.listdir(PATH_IMAGES):
        genre_path = os.path.join(PATH_IMAGES, dir_genre)
        genre_imgs[dir_genre] = loader.load_img(genre_path)

    """
    estructura del diccionari:
    images = {
    'pop': { 'pop00017.png': np.array(...), 'pop00018.png': np.array(...) },
    'rock': { 'rock00021.png': np.array(...), 'rock00022.png': np.array(...) },}
    """

    print("\n--- PRE-PROCESSAR ---")
    data3 = DataPreprocessor()
    data3.preprocess_csv(df_3s)
    data3.split_data()
    data3.normalize_data()
    data3.plot_features('data3_features.png')
    # data3.remove_noise()   #!!!! provar amb diferents thresholds, per decidir quin es el millor
    print(f"CSV 3s carregat. Train shape: {data3.train_data.shape}, Test shape: {data3.test_data.shape}")

    data30 = DataPreprocessor()
    data30.preprocess_csv(df_30s)
    data30.split_data()
    data30.normalize_data()
    data30.plot_features('data30_features.png')
    # data30.remove_noise()
    print(f"CSV 30s carregat. Train shape: {data30.train_data.shape}, Test shape: {data30.test_data.shape}")

    dataIMG = DataPreprocessor()
    dataIMG.preprocess_images(genre_imgs)
    dataIMG.remove_noise()
    dataIMG.split_data()
    X_images = dataIMG._X # imatges processades
    y_labels = dataIMG._y # etiquetes codificades
    print(f"IMAGES carregades. Train shape: {dataIMG.train_data.shape}, Test shape: {dataIMG.test_data.shape}")



    print("\n--- IMPLEMENTAR MODELS ---")
    models3 = Models(data3.train_data, data3.train_labels, data3.test_data, data3.test_labels)
    dataset_name = 'df_3s'
    MODELS3_DICT = get_models(models3)
    LABELS = data3.get_labels()

    #* Entrenem cada model
    for model_str, model_train in MODELS3_DICT.items():
        model_train(dataset_name) 
        models3.evaluate_model(model_str, dataset_name, LABELS)

    metrics3_df = models3.create_metrics_dataframe()
    #models3.do_plot_metrics('metrics.csv')
    
    """ NO FUNCIONA
    model_file = os.path.join(models3._cache, f"{model_str.replace(' ', '')}_{dataset_name}.pkl")
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        models3.generate_roc_curve(models3.generate_roc_curve(model, data3.test_data, data3.test_labels, model_str))
    """

    models30 = Models(data30.train_data, data30.train_labels, data30.test_data, data30.test_labels)
    dataset_name = 'df_30s'
    MODELS30_DICT = get_models(models30)

    for model_str, model_train in MODELS30_DICT.items():
        model_train(dataset_name)
        models30.evaluate_model(model_str, dataset_name, LABELS)

    metrics30_df = models30.create_metrics_dataframe()
    #models30.do_plot_metrics('metrics.csv')
    
    model_file = os.path.join(models30._cache, f"{model_str.replace(' ', '')}_{dataset_name}.pkl")
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        models30.generate_roc_curve(model, data30.test_data, data30.test_labels, model_str)

    #* FEM UN MERGE PER TENIR UN CSV UNIC DE LES DADES DE 3 I 30S
    merged_df = pd.concat([metrics3_df, metrics30_df], ignore_index=True)
    merged_df.to_csv('metrics.csv', index=False)

if __name__ == "__main__":
    main()