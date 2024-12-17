from loadData import DataLoader
from dataPreprocessor import DataPreprocessor
from algorithms import Models
from evaluateParams import HyperparameterEvaluator
import os
import pandas as pd


def get_models(model: object) -> dict:
    return {
        "KNN": model.do_knn,
        "SVM": model.do_svm,
        "Decision Tree": model.do_decision_tree,
        "Random Forest": model.do_random_forest,
        "Gradient Boosting": model.do_gradient_boosting, #! --> LO COMENTO PARA LAS IMAGES PQ TARDA MUCHO
        "Logistic Regression": model.do_logistic_regression,
        "Gaussian NB": model.do_gaussian_naive_bayes,
        "Bernoulli NB": model.do_bernoulli_naive_bayes,
        "Multinomial NB": model.do_multinomial_nb,
    }


def main():
    PATH_CSV3 = r"data/features_3_sec.csv"
    PATH_CSV30 = r"data/features_30_sec.csv"
    PATH_IMAGES = r"data/images_original"
    PATH_AUDIOS = r"data/genres_original"

    print(
        "----------------------------- CARREGAR DADES ---------------------------------"
    )
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

    print(
        "\n--------------------------- PRE-PROCESSAR ---------------------------------"
    )
    data3 = DataPreprocessor()
    data3.preprocess_csv(df_3s)
    data3.split_data()
    data3.plot_features("data3_features.png")
    data3.normalize_data()
    # data3.remove_noise()   #!!!! provar amb diferents thresholds, per decidir quin es el millor
    print(f"CSV 3s carregat. Train shape: {data3.train_data.shape}, Test shape: {data3.test_data.shape}")

    data30 = DataPreprocessor()
    data30.preprocess_csv(df_30s)
    data30.split_data()
    data30.plot_features("data30_features.png")
    data30.normalize_data()
    # data30.remove_noise()
    print(f"CSV 30s carregat. Train shape: {data30.train_data.shape}, Test shape: {data30.test_data.shape}")

    dataIMG = DataPreprocessor()
    dataIMG.preprocess_images(genre_imgs)
    dataIMG.remove_noise()
    dataIMG.split_data()

    # * Hem de redimensionar l'array de les imatges perque Machine Learning models accepts 2D arrays
    dataIMG.train_data = dataIMG.train_data.reshape(dataIMG.train_data.shape[0], -1)
    dataIMG.test_data = dataIMG.test_data.reshape(dataIMG.test_data.shape[0], -1)

    X_images = dataIMG._X  # imatges processades
    y_labels = dataIMG._y  # etiquetes codificades
    print(f"IMAGES carregades. Train shape: {dataIMG.train_data.shape}, Test shape: {dataIMG.test_data.shape}")

    print(
        "\n------------------------------ IMPLEMENTAR MODELS ---------------------------------"
    )
    print("--- AMB CSV ---")
    models3 = Models(data3.train_data, data3.train_labels, data3.test_data, data3.test_labels)

    dataset_name = "df_3s"
    MODELS3_DICT = get_models(models3)
    LABELS = data3.get_labels()

    # * Entrenem cada model
    for model_str, model_train in MODELS3_DICT.items():
        model_train(dataset_name)
        models3.evaluate_model(model_str, dataset_name, LABELS)

    metrics3_df = models3.create_metrics_dataframe()
    models3.do_plot_metrics(suffix="_3s")

    models30 = Models(data30.train_data, data30.train_labels, data30.test_data, data30.test_labels)
    dataset_name = "df_30s"
    MODELS30_DICT = get_models(models30)
    LABELS = data30.get_labels()

    for model_str, model_train in MODELS30_DICT.items():
        model_train(dataset_name)
        models30.evaluate_model(model_str, dataset_name, LABELS)

    metrics30_df = models30.create_metrics_dataframe()
    models30.do_plot_metrics(suffix="_30s")

    print("\n---AMB IMATGES ---")

    modelsIMG = Models(dataIMG.train_data, dataIMG.train_labels, dataIMG.test_data, dataIMG.test_labels)
    dataset_name = "images"
    MODELSIMG_DICT = get_models(modelsIMG)
    LABELS = dataIMG.get_labels()

    for model_str, model_train in MODELSIMG_DICT.items():
        print(f"Entrenant el model {model_str} amb les imatges...")
        model_train(dataset_name)
        modelsIMG.evaluate_model(model_str, dataset_name, LABELS)

    metricsIMG_df = modelsIMG.create_metrics_dataframe("metrics_images.csv")
    print(f"Mètriques dels models amb imatges desades a 'metrics_images.csv'")

    modelsIMG.do_plot_metrics(suffix="images", metrics_filename="metrics_images.csv")

    # * FEM UN MERGE PER TENIR UN CSV UNIC DE LES DADES DE 3 I 30S
    merged_df = pd.concat([metrics3_df, metrics30_df], ignore_index=True)
    merged_df.to_csv("metrics.csv", index=False)

    print(
        "\n------------------------------ AJUSTAR HIPERPÀREMTRES ---------------------------------"
    )
    PICKLE_DIR = "cache_data"
    
    datasets = {
        "data3": {"X": data3.train_data, "labels": data3.train_labels},
        "data30": {"X": data30.train_data, "labels": data30.train_labels},
        "dataIMG": {"X": dataIMG.train_data, "labels": dataIMG.train_labels},
    }

    # Definició dels hiperparàmetres de tots els models --> gradient boosting tarda mucho
    """ TARDA MUCHISIMO
    "Gradient Boosting": {
        "learning_rate": [0.5, 1, 5],
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7]
    },
    """
    MODELS_PARAMS = {
        "Random Forest": {
            "n_estimators": [50, 100],
            "max_depth": [2, 10, 20],
            "min_samples_split": [2, 5]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "kd_tree"]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": [1, 4, 7, 11],
            "degree": [2, 3]  # per al kernel "poly"
        },
        "Gaussian NB": {
            "var_smoothing": [1e-9, 1e-8, 1e-7]  # control de la suavització
        },
        "Logistic Regression": {
                "C": [0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["saga", "liblinear"] #!!!! si no ejecuta eliminar el saga
            }
    }
    
    # models -- datasets amb pickle files
    param_grids = {
        "data3": {
            "Random Forest": ("RandomForest_df_3s.pkl", MODELS_PARAMS["Random Forest"]),
            "KNN": ("KNN_df_3s.pkl", MODELS_PARAMS["KNN"]),
            "SVM": ("SVM_df_3s.pkl", MODELS_PARAMS["SVM"])
        },
        "data30": {
            "Random Forest": ("RandomForest_df_30s.pkl", MODELS_PARAMS["Random Forest"]),
            "KNN": ("KNN_df_30s.pkl", MODELS_PARAMS["KNN"]),
            "SVM": ("SVM_df_30s.pkl", MODELS_PARAMS["SVM"])
        },
        "dataIMG": {
            "Random Forest": ("RandomForest_images.pkl", MODELS_PARAMS["Random Forest"]),
            "Gaussian NB": ("GaussianNB_images.pkl", MODELS_PARAMS["Gaussian NB"]),
            "Logistic Regression": ("LogisticRegression_images.pkl", MODELS_PARAMS["Logistic Regression"]),
        }
    }
    
    for key, dataset in datasets.items():
        print(f"\n--- Processant hiperparàmetres per a {key} ---")
        evaluator = HyperparameterEvaluator(PICKLE_DIR, dataset["X"], dataset["labels"])
        evaluator.process_models(param_grids[key])


if __name__ == "__main__":
    main()