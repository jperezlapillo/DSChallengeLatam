import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import seaborn as sns

def get_tasa_atraso(df, var, tasa_agg):
    """
    Genera gráfico con tasa de atraso (en %) para una variable específica
    Incluye valor promedio agregado para comparaciones
    :param df: dataframe
    :param var: variable a analizar
    :param tasa_agg: tasa agregada de atrasos
    :return: figura 
    """
    plt.figure()
    result = round(df.groupby(var)["atraso_15"].sum() / df.groupby(var)["SIGLAORI"].count()*100,1)
    fig = result.plot(kind="bar", figsize=(12,4))
    plt.axhline(y=tasa_agg, color='purple', linestyle='dotted')
    plt.title("Tasa de atraso por {}".format(var))
    
    return plt.show(fig)



def get_best_params_results(model, 
                            param_grid,
                            train_feat,
                            train_label,
                            test_feat,
                            test_label,
                            scoring="balanced_accuracy"):
    """
    Realiza entrenamiento para un modelo, tomando un grid search de parámetros
    y entrega resultados para el mejor modelo según la métrica definida
    :param model: modelo clasificador genérico
    :param param_grid: grilla de parámetros sobre los cuales se va a testear
    :param train_feat: features del training set
    :param train_label: etiquetas del training set
    :param test_feat: features del test set
    :param test_label: etiquetas del test set
    :param scoring: métrica, por default "balanced_accuracy"
    :return: mmejor odelo clasificador y su métrica asociada
    """
    classifier = GridSearchCV(estimator=model, 
                              param_grid=param_grid, 
                              scoring=scoring).fit(train_feat, train_label)
    
    predictions = classifier.predict(test_feat)
    accuracy = balanced_accuracy_score(test_label, predictions)
    
    return classifier, accuracy

def plot_matrix(cm, classes, title):
    """
    Genera gráfico de matriz de confusión
    :param cm: matriz de confusión (array)
    :param classes: lista de clases del problema
    :param title: título del gráfico
    :return: figura
    """
    ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt="g", xticklabels=classes, yticklabels=classes, cbar=False)
    ax.set(title=title, xlabel="Predicted label", ylabel="True label")
    
    return plt.show(ax)

