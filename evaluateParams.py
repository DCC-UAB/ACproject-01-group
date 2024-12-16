import os
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HyperparameterEvaluator:
    def __init__(self, pickle_dir, X_train, y_train, scoring="accuracy", cv=3, n_jobs=12) -> None:
        self._pickle_dir = pickle_dir
        self._X_train = X_train
        self._y_train = y_train
        self._scoring = scoring # Mètrica per optimitzar.
        self._cv = cv # Número de particions per cross-validation.
        self._n_jobs = n_jobs

    def load_model(self, pickle_file):
        """
        Carrega un model des d'un arxiu pickle. I retorna el model.
        """
        filepath = os.path.join(self._pickle_dir, pickle_file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"El fitxer {pickle_file} no existeix al directori {self._pickle_dir}.")
        
        with open(filepath, "rb") as f:
            print("Arxiu:", filepath)
            return pickle.load(f)

    def evaluate_hyperparameters(self, model: object, param_grid: dict, search_type="grid") -> None:
        """
        Avalua i ajusta els hiperparàmetres d'un model donat.
        :param model: Model a avaluar.
        :param param_grid: Diccionari amb els hiperparàmetres a provar.
        :param search_type: Tipus de cerca: 'grid' o 'random'.
        :return: Resultats de la millor configuració d'hiperparàmetres.
        """
        if search_type == "grid":
            search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self._scoring, cv=self._cv, n_jobs=self._n_jobs)
        elif search_type == "random": #??????? FUNCIONARIA GRADIENTBOOSTING
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=self._scoring, cv=self._cv, n_jobs=self._n_jobs, n_iter=50)
        else:
            raise ValueError("search_type ha de ser 'grid' o 'random'.")
        
        print("Ajustant hiperparàmetres...")
        search.fit(self._X_train, self._y_train)
        
        print(f"Millors hiperparàmetres trobats: {search.best_params_}")
        print(f"Millor puntuació aconseguida: {search.best_score_}")

    def process_models(self, models_and_grids, search_type="grid") -> None:
        """
        Processa múltiples models amb els seus respectius param_grid.
        :param models_and_grids: {nom_model: (pickle_file, param_grid)}.
        :param search_type: Tipus de cerca: 'grid' o 'random'.
        """
        for model_name, (pickle_file, param_grid) in models_and_grids.items():
            print(f"\nProcessant model: {model_name}")
            model = self.load_model(pickle_file)
            self.evaluate_hyperparameters(model, param_grid, search_type)