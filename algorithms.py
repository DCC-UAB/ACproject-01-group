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

        self._prediccions = {}  # ? {nom_model : predict}
        self._metrics = []  # emmagatzema les mètriques de RENDIMENT dels models, per fer el dataframe
        self._cache = "cache_data"

    ######### SAVE
    def create_metrics_dataframe(self, filename="metrics.csv") -> pd.DataFrame:
        """Crear un DataFrame amb les mètriques per a cada model.
        També crea un csv"""
        # * Crear el DataFrame
        metrics_df = pd.DataFrame(self._metrics)

        # * Desar el DataFrame en un fitxer CSV
        metrics_df.to_csv(filename, index=False)
        print(f"Mètriques desades a {filename}")

        return metrics_df

    def save_model(self, model_name: str, model: object, filename: str):
        """ENTRENAR i GUARDAR el model si no existeix."""
        # Comprovar si el model ja existeix com a pickle
        if os.path.exists(filename):
            print(f"El model {model_name} ja està entrenat i guardat.")
            with open(filename, "rb") as f:
                model = pickle.load(f)
        else:
            print(f"Entrenant el model {model_name}...")
            model.fit(self._X_train, self._y_train)
            with open(filename, "wb") as f:
                pickle.dump(model, f)
            print(f"Model {model_name} entrenat i guardat com a pickle.")

        y_pred = model.predict(self._X_test)
        self._prediccions[model_name] = y_pred

    ######### ALGORISMES - MODELS
    def do_knn(self, dataset_name: str, n_neighbors=5, weights="uniform", algorithm="auto"):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        filename = os.path.join(self._cache, f"KNN_{dataset_name}.pkl")
        self.save_model("KNN", knn, filename)

    def do_svm(self, dataset_name: str, kernel="rbf", C=1.0, gamma="scale", random_state=None):
        svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
        filename = os.path.join(self._cache, f"SVM_{dataset_name}.pkl")
        self.save_model("SVM", svm, filename)

    def do_decision_tree(self, dataset_name: str, random_state=42):
        dtree = DecisionTreeClassifier(random_state=random_state)
        filename = os.path.join(self._cache, f"DecisionTree_{dataset_name}.pkl")
        self.save_model("Decision Tree", dtree, filename)

    def do_random_forest(self, dataset_name: str, n_estimators=100, random_state=42):
        random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        filename = os.path.join(self._cache, f"RandomForest_{dataset_name}.pkl")
        self.save_model("Random Forest", random_forest, filename)

    def do_gradient_boosting(self, dataset_name: str, learning_rate=0.3, n_estimators=20, random_state=42):
        gb = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)
        filename = os.path.join(self._cache, f"GradientBoosting_{dataset_name}.pkl")
        self.save_model("Gradient Boosting", gb, filename)

    def do_logistic_regression(self, dataset_name: str, C=1.0, solver="liblinear", max_iter=5000, penalty="l2", random_state=42,):
        logistic_regression = LogisticRegression(C=C, solver=solver, max_iter=max_iter, penalty=penalty, random_state=random_state)
        filename = os.path.join(self._cache, f"LogisticRegression_{dataset_name}.pkl")
        self.save_model("Logistic Regression", logistic_regression, filename)

    def do_gaussian_naive_bayes(self, dataset_name: str):
        gnb = GaussianNB()
        filename = os.path.join(self._cache, f"GaussianNB_{dataset_name}.pkl")
        self.save_model("Gaussian NB", gnb, filename)

    def do_bernoulli_naive_bayes(self, dataset_name: str, alpha=1.0, binarize=0):
        bnb = BernoulliNB(alpha=alpha, binarize=binarize)
        filename = os.path.join(self._cache, f"BernoulliNB_{dataset_name}.pkl")
        self.save_model("Bernoulli NB", bnb, filename)

    def do_multinomial_nb(self, dataset_name: str, alpha=1.0):
        multinomial_nb = MultinomialNB(alpha=alpha)
        filename = os.path.join(self._cache, f"MultinomialNB_{dataset_name}.pkl")
        self.save_model("Multinomial NB", multinomial_nb, filename)

    ######## METRIQUES
    def do_confusion_matrix(self, cm: object, model_name: str, dataset_name: str, labels: list, dir="confusion_matrixs", show=False) -> None:
        """Visualitzar la matriu de confusió."""
        plt.figure(figsize=(8, 6))

        # Definim els "eixos" amb NOM dels labels
        if labels:
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
        else:
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")

        plt.title(f"Matriu de Confusió - {model_name} - {dataset_name}")
        plt.xlabel("Predicció")
        plt.ylabel("Realitat")

        # Si el directori no existeix, es crea
        if not os.path.exists(dir):
            os.makedirs(dir)

        # desar com imatge
        output_filename = os.path.join(dir, f"{model_name}_{dataset_name}_confusion_matrix.png")
        plt.savefig(output_filename)
        print(f"Matriu de confusió desada a {output_filename}")

        if show:
            plt.show()

    def evaluate_model(self, model_name: str, dataset_name: str, labels: list) -> None:
        """Avalua un model determinat i l'afegeix a la llista per crear posteriorment el dataset"""
        if model_name not in self._prediccions:
            raise ValueError(f"El model '{model_name}' no es troba en el diccionari.")

        prediction = self._prediccions[model_name]

        # Avaluem el model amb les metriques
        accuracy = accuracy_score(self._y_test, prediction)
        precision = precision_score(self._y_test, prediction, average="weighted")
        f1 = f1_score(self._y_test, prediction, average="weighted")
        cm = confusion_matrix(self._y_test, prediction, normalize="true")

        # Visualitzar la matriu de confusió
        self.do_confusion_matrix(cm, model_name, dataset_name, labels)

        # Afegir resultats al registre
        self._metrics.append({
                    "Algorisme": model_name,
                    "Dataset": dataset_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "F1-Score": f1
                })

        # Generar curva ROC
        try:
            self.plot_roc_curve(model_name, dataset_name, labels)
        except ValueError as e:
            print(f"Error al generar la curva ROC para {model_name}: {e}")

    def do_plot_metrics(self, suffix, metrics_filename="metrics.csv", output_dir="plot_metrics", show=False) -> None:
        metrics_df = pd.read_csv(metrics_filename)
        models = metrics_df["Algorisme"]
        metrics = metrics_df[["Accuracy", "Precision", "F1-Score"]]

        # Crear el grafic de barres
        fig, ax = plt.subplots(figsize=(15, 6))
        n_models = len(models)
        n_metrics = len(metrics.columns)

        # Configrar les posicions de les barres
        bar_width = 0.1
        x = range(n_models)
        colors = ["pink", "lightgreen", "lightblue"]  # Colors per cada metrica

        # Dibuixem les barres per cada metrica
        for idx, metric in enumerate(metrics.columns):
            bar_positions = [
                pos + idx * bar_width for pos in x
            ]  # Desplazar les posicions pq quadri
            ax.bar(
                bar_positions,
                metrics[metric],
                width=bar_width,
                label=metric,
                color=colors[idx],
            )

        # Ajustar les posicions en l'eix de les X perque esten centrades
        ax.set_xticks([pos + bar_width * (n_metrics - 1) / 2 for pos in x])
        ax.set_xticklabels(models)

        ax.set_title(f"Comparació de Metriques dels Models {suffix}", fontsize=16)
        ax.set_xlabel("Models", fontsize=14)
        ax.set_ylabel("Valor de Metriques", fontsize=14)
        ax.set_ylim(0,1) #!!! definim el rang 
        ax.legend(title="Metriques", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        # Crear la carpeta si no existeix i guardar el gràfic
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, f"metrics_plot{suffix}.png")
        plt.savefig(output_filename)
        print(f"Visualització de mètriques desada a {output_filename}")

        if show:
            plt.show()

    def plot_roc_curve(self, model_name: str, dataset_name: str, labels: list, output_dir = "roc_curves") -> None:
        """Generar la corba ROC per a un model determinat"""
        if model_name not in self._prediccions:
            raise ValueError(f"El model '{model_name}' no es troba en el diccionari.")

        prediction = self._prediccions[model_name]

        # Binaritzar les etiquetes de sortida per a la corba ROC
        y_test_binarized = label_binarize(self._y_test, classes=labels)
        n_classes = y_test_binarized.shape[1]

        # Comprovar si la predicció és unidimensional (classificació binària)
        if prediction.ndim == 1:
            prediction = label_binarize(prediction, classes=labels)

        # Calcular la corba ROC i l'àrea sota la corba (AUC) per a cada classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], prediction[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Dibuixar la corba ROC
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"Classe {labels[i]} (àrea = {roc_auc[i]:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Taxa de Falsos Positius")
        plt.ylabel("Taxa de Veritables Positius")
        plt.title(f"Corba ROC - {model_name} - {dataset_name}")
        plt.legend(loc="lower right")

        # Desar el gràfic
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_roc_curve.png")
        plt.savefig(output_path)
        print(f"Corba ROC desada a {output_path}")
