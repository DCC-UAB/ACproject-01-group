"""
FUNCIÓ PRINCIPAL:

"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle

class Models:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

        self._prediccions = {}  #? {nom_model : predict}
        self._metrics = []  # emmagatzema les mètriques de RENDIMENT dels models, per fer el dataframe
        self._cache = 'cache_data'

    ######### SAVE
    def create_metrics_dataframe(self, filename='metrics.csv') -> pd.DataFrame:
        """Crear un DataFrame amb les mètriques per a cada model.
        També crea un csv """
        #* Crear el DataFrame
        metrics_df = pd.DataFrame(self._metrics)
        
        #* Desar el DataFrame en un fitxer CSV
        metrics_df.to_csv(filename, index=False)
        print(f"Mètriques desades a {filename}")
        
        return metrics_df
    
    def save_model(self, model_name:str, model:object, filename:str):
        """ENTRENAR i GUARDAR el model si no existeix."""
        # Comprovar si el model ja existeix com a pickle
        if os.path.exists(filename):
            print(f"El model {model_name} ja està entrenat i guardat.")
            with open(filename, 'rb') as f:
                model = pickle.load(f)
        else:
            print(f"Entrenant el model {model_name}...")
            model.fit(self._X_train, self._y_train)
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model {model_name} entrenat i guardat com a pickle.")

        y_pred = model.predict(self._X_test)
        self._prediccions[model_name] = y_pred
    

    ######### ALGORISMES - MODELS
    def do_decision_tree(self, dataset_name:str, random_state=42):
        dtree = DecisionTreeClassifier(random_state=random_state)
        filename = os.path.join(self._cache, f'DecisionTree_{dataset_name}.pkl')
        self.save_model('Decision Tree', dtree, filename)
    
    def do_random_forest(self, dataset_name:str, n_estimators=100, random_state=42):
        random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        filename = os.path.join(self._cache, f'RandomForest_{dataset_name}.pkl')
        self.save_model('Random Forest', random_forest, filename)

    def do_gradient_boosting(self, dataset_name:str, learning_rate=0.1, n_estimators=100, random_state=42):
        gb = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)
        filename = os.path.join(self._cache, f'GradientBoosting_{dataset_name}.pkl')
        self.save_model('Gradient Boosting', gb, filename)

    def do_logistic_regression(self, dataset_name:str, C=1.0,solver='liblinear', max_iter=5000, penalty='l2', random_state=42):
        logistic_regression = LogisticRegression(C=1, solver='liblinear', max_iter=5000, penalty='l2', random_state=42)
        filename = os.path.join(self._cache, f'LogisticRegression{dataset_name}.pkl')
        self.save_model('Logistic Regression', logistic_regression, filename)
        
    ######## METRIQUES
    def do_confusion_matrix(self, cm:object, model_name:str, dataset_name:str, dir='confusion_matrixs', show=False):
        """Visualitzar la matriu de confusió."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"Matriu de Confusió - {model_name} - {dataset_name}")
        plt.xlabel("Predicció")
        plt.ylabel("Realitat")

        #* Desar la matriu com a imatge
        # Si el directori no existeix, es crea
        if not os.path.exists(dir):
            os.makedirs(dir)
        output_filename = os.path.join(dir, f"{model_name}_{dataset_name}_confusion_matrix.png")
        plt.savefig(output_filename)
        print(f"Matriu de confusió desada com a {output_filename}")

        if show:
            plt.show()

    def evaluate_model(self, model_name:str, dataset_name:str):
        """Avalua un model determinat i l'afegeix a la llista per crear posteriorment el dataset"""
        if model_name not in self._prediccions:
            raise ValueError(f"El model '{model_name}' no es troba en el diccionari.")
        
        prediction = self._prediccions[model_name]

        # Avaluem el model amb les metriques
        accuracy = accuracy_score(self._y_test, prediction)
        precision = precision_score(self._y_test, prediction, average="weighted")
        f1 = f1_score(self._y_test, prediction, average="weighted")
        cm = confusion_matrix(self._y_test, prediction, normalize='true')

        # Visualitzar la matriu de confusió
        self.do_confusion_matrix(cm, model_name, dataset_name)

        # Afegir resultats al registre
        self._metrics.append({
            "Algorisme": model_name,
            "Dataset": dataset_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "F1-Score": f1
        })

    def do_plot_metrics(self, metrics_filename, show=True):
        #TODO: fer que indiqui cada model 
        metrics_df = pd.read_csv(metrics_filename)
        models = metrics_df["Algorisme"]
        metrics = metrics_df[["Accuracy", "Precision", "F1-Score"]]
        
        # Crear el grafic de barres
        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = None
        colors = ["pink", "lightgreen", "lightblue"]  # Colores para cada métrica

        # Iterem sobre les metriques 
        for idx, metric in enumerate(metrics.columns):
            if bottom is None:
                ax.bar(models, metrics[metric], label=metric, color=colors[idx])
                bottom = metrics[metric]
            else:
                ax.bar(models, metrics[metric], bottom=bottom, label=metric, color=colors[idx])
                bottom += metrics[metric]

        ax.set_title("Comparació de Metriques dels MOdels", fontsize=16)
        ax.set_xlabel("Models", fontsize=14)
        ax.set_ylabel("Valor de Metriques", fontsize=14)
        ax.legend(title="Metriques", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig("plot_metrics.png")
            print("Grafic desat com plot_metrics.png'")


    

