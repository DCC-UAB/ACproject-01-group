"""
FUNCIÓ PRINCIPAL:
Aquest script serveix per descriure els diferents algoritmes per al nostre model. 
A més d'implementar diferents metodes per visualitzar els nostres resultats. 
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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
    def do_knn(self, dataset_name:str, n_neighbors=5, weights='uniform', algorithm='auto'):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        filename = os.path.join(self._cache, f'KNN_{dataset_name}.pkl')
        self.save_model('KNN', knn, filename)

    def do_svm(self, dataset_name:str, kernel='rbf', C=1.0, gamma='scale', random_state=None):
        svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
        filename = os.path.join(self._cache, f'SVM_{dataset_name}.pkl')
        self.save_model('SVM', svm, filename)

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
        filename = os.path.join(self._cache, f'LogisticRegression_{dataset_name}.pkl')
        self.save_model('Logistic Regression', logistic_regression, filename)
    
    def do_gaussian_naive_bayes(self, dataset_name:str):
        gnb = GaussianNB()
        filename = os.path.join(self._cache, f'GaussianNB_{dataset_name}.pkl')
        self.save_model('Gaussian NB', gnb, filename)

    def do_bernoulli_naive_bayes(self, dataset_name:str, alpha=1.0, binarize=0):
        bnb = BernoulliNB(alpha=alpha, binarize=binarize)
        filename = os.path.join(self._cache, f'BernoulliNB_{dataset_name}.pkl')
        self.save_model('Bernoulli NB', bnb, filename)

    def do_multinomial_nb(self, dataset_name: str, alpha=1.0):
        multinomial_nb = MultinomialNB(alpha=alpha)
        filename = os.path.join(self._cache, f'MultinomialNB_{dataset_name}.pkl')
        self.save_model('Multinomial NB', multinomial_nb, filename)


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
        print(f"Matriu de confusió desada a {output_filename}")

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

    def generate_roc_curve(model, X_test, y_test, model_name, output_path="roc_curves"):
        """Genera y guarda la curva ROC para un modelo de clasificación.
        Compatible con modelos que tienen o no 'predict_proba'."""
        
        # Verificar si las etiquetas son continuas
        if np.issubdtype(np.array(y_test).dtype, np.floating):
            print("Error: Las etiquetas son continuas. ROC no es aplicable.")
            return  # Detener el proceso si las etiquetas son continuas

        # Binarizar etiquetas para problemas multiclase
        y_test = np.array(y_test).flatten()
        n_classes = len(set(y_test))
        y_test_binarized = label_binarize(y_test, classes=list(range(n_classes)))

        try:
            y_prob = model.predict_proba(X_test)
        except AttributeError:
            print(f"{model_name} no tiene 'predict_proba'. Usando 'predict' en su lugar.")
            y_prob = model.predict(X_test)
            y_prob = np.expand_dims(y_prob, axis=1)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        if y_prob.ndim > 1:  
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr_micro, tpr_micro, _ = roc_curve(y_test_binarized.ravel(), y_prob.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)

            plt.figure(figsize=(10, 8))
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], label=f"Clase {i} (AUC = {roc_auc[i]:.2f})")
            
            plt.plot(fpr_micro, tpr_micro, label=f"Micro promedio (AUC = {roc_auc_micro:.2f})", linestyle='--', color='red')

        else:
            fpr[0], tpr[0], _ = roc_curve(y_test_binarized[:, 0], y_prob)
            roc_auc[0] = auc(fpr[0], tpr[0])

            plt.figure(figsize=(10, 8))
            plt.plot(fpr[0], tpr[0], label=f"Clase 0 (AUC = {roc_auc[0]:.2f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label="Aleatoria (AUC = 0.5)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Tasa de Falsos Positivos (FPR)")
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
        plt.title(f"Curva ROC - {model_name}")
        plt.legend(loc="lower right")

        output_file = os.path.join(output_path, f"roc_curve_{model_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(plt, f)
        plt.close()

        print(f"Curva ROC guardada en {output_file}")