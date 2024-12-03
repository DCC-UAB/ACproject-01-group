"""
FUNCIÓ PRINCIPAL:

"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Models:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

        self._prediccions = {} #??? {nom_model : predict}
        self._metrics = [] # llista per emmagatzemar les metriques fetes amb cada algoritme

    def create_metrics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._metrics)
    
    
    def do_decision_tree(self, random_state=42):
        dtree = DecisionTreeClassifier(random_state=random_state)
        dtree.fit(self._X_train, self._y_train)
        y_pred_tree = dtree.predict(self._X_test)
        self._prediccions['Decision Tree'] = y_pred_tree
    
    def do_random_forest(self, n_estimators=100, random_state=42):
        """!!!!
        TODO ESTO ESTA ORDINAL ENCODED EN LA PRACTICA DE CLASE, EL TRAIN Y TEST --- TODAS 
        !!!!!"""
        random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        random_forest.fit(self._X_train, self._y_train)
        y_pred_forest = random_forest.predict(self._X_test)
        self._prediccions['Random Forest'] = y_pred_forest 

    def do_gradient_boosting(self, learning_rate=0.1, n_estimators=100, random_state=42):
        """
        Crea i entrena un model Gradient Boosting.
        """
        gb = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)
        gb.fit(self._X_train, self._y_train)
        y_pred_gb = gb.predict(self._X_test)
        self._prediccions['Gradient Boosting'] = y_pred_gb  
    

    ######## METRIQUES
    def do_confusion_matrix(self, cm, model_name):
        """
        Visualitza la matriu de confusió.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues") #?? xticklabels=unique_labels, yticklabels=unique_labels
        plt.title(f"Matriu de Confusió - {model_name}")
        plt.xlabel("Predicció")
        plt.ylabel("Realitat")
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
        
        #TODO: self.do_confusion_matrix(cm, model_name) ---> PER FER EL PLOT.

        # Afegir resultats al registre
        self._metrics.append({
            "Algorisme": model_name,
            "Dataset": dataset_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "F1-Score": f1
        })