"""
FUNCIÓ PRINCIPAL:

"""
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class Models:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

        self._prediccions = {}  # Diccionari per emmagatzemar les prediccions dels models
        self._metrics = []  # Llista per emmagatzemar les mètriques dels models
        self._cache = 'cache_data'

    def create_metrics_dataframe(self, filename='metrics.csv') -> pd.DataFrame:
        """Crear un DataFrame amb les mètriques per a cada model.
        També crea un csv """
        #* Crear el DataFrame
        metrics_df = pd.DataFrame(self._metrics)
        
        #* Desar el DataFrame en un fitxer CSV
        metrics_df.to_csv(filename, index=False)
        print(f"Mètriques desades a {filename}")
        
        return metrics_df
    
    def save_model(self, model_name, model, filename):
        """Entrenar i guardar el model si no existeix."""
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
    
    def do_decision_tree(self, dataset_name, random_state=42):
        dtree = DecisionTreeClassifier(random_state=random_state)
        filename = os.path.join(self._cache, f'DecisionTree_{dataset_name}.pkl')
        self.save_model('Decision Tree', dtree, filename)
    
    def do_random_forest(self, dataset_name, n_estimators=100, random_state=42):
        random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        filename = os.path.join(self._cache, f'RandomForest_{dataset_name}.pkl')
        self.save_model('Random Forest', random_forest, filename)

    def do_gradient_boosting(self, dataset_name,learning_rate=0.1, n_estimators=100, random_state=42):
        gb = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)
        filename = os.path.join(self._cache, f'GradientBoosting_{dataset_name}.pkl')
        self.save_model('Gradient Boosting', gb, filename)

    ######## METRIQUES
    def do_confusion_matrix(self, cm, model_name, dataset_name, dir='confusion_matrixs', show=False):
        """Visualitzar la matriu de confusió."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"Matriu de Confusió - {model_name} - {dataset_name}")
        plt.xlabel("Predicció")
        plt.ylabel("Realitat")

        #* Desar la matriu com a imatge
        # Comprovar si el directori existeix, si no, crear-lo
        if not os.path.exists(dir):
            os.makedirs(dir)
        output_filename = os.path.join(dir, f"{model_name}_{dataset_name}_confusion_matrix.png")
        plt.savefig(output_filename)
        print(f"Matriu de confusió desada com a {output_filename}")

        if show:
            plt.show()

    def evaluate_model(self, model_name, dataset_name):
        """Evalua un model determinat i l'afegeix a la llista per crear posteriorment el dataset"""
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