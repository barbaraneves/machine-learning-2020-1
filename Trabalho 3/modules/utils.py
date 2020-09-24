import numpy as np 
import pandas as pd
import random

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from modules import models

#Funções do Trabalho 2
def plot_confusion_matrix(y_true, y_pred, title, cmap=plt.cm.Reds):
    cm = confusion_matrix(y_true, y_pred) #Computar a matrix de confusão
    classes = [int(i) for i in np.unique(y_true)] #Classes

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, 
           yticklabels=classes,
           title=title,
           ylabel='Verdadeiros',
           xlabel='Preditos')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax

def plot_boundaries(X, y, clf, title, cmap=plt.cm.YlOrRd):
    markers = ('o', 'x')
    colors = ('firebrick', 'black')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = np.array(Z).reshape(xx.shape)
    
    plt.figure(figsize=(8, 5))
    plt.title(label=title)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1], 
                    c=colors[idx],
                    marker=markers[idx], 
                    label='Class ' + str(int(cl)), 
                    edgecolor='black')
    plt.legend()
    plt.show()

def accuracy_score(y_real, y_pred):
    return np.sum(y_pred == y_real)/y_real.shape[0]

# Função extra do Trabalho 2
def plot_loss_path(loss, title=None):
    plt.figure(figsize=(10, 5))
    
    plt.rcParams.update({'font.size': 14})
    plt.plot(range(1, len(loss)+1), loss, '-k', color='firebrick')
    
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    if title is not None:
        plt.title(title, fontsize=14)        
        
    plt.show()

#Novas funções
def plot_data(X, y, marker='o', cmap='YlOrRd', title=False):
    classes = [int(i) for i in np.unique(y)] 
    cm = plt.get_cmap(cmap)
    colors=[cm(1.*i/20) for i in range(20)]
    
    fig = plt.figure(figsize=(8, 6), )
    plt.rcParams.update({'font.size': 14})
    
    for i, class_ in enumerate(classes):       
        plt.scatter(X[y == class_, 0], X[y == class_, 1], s=10*12, marker=marker, color=colors[(i+1)*9])
    
    if title:
        plt.title(label=title)
        
    plt.show()
    xlim = fig.gca().get_xlim() 
    ylim = fig.gca().get_ylim()

# 3ª questão: K-fold
def k_fold(X, y, k, method, seed=42):
    idx = list(range(len(X)))
    subset_size = round(len(X)/k)
    metric_values = []
    
    random.Random(seed).shuffle(idx)
    subsets = [idx[X:X + subset_size] for X in range(0, len(idx), subset_size)]
    
    for i in range(k):
        X_ = X[subsets[i]]
        y_ = y[subsets[i]]
        
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=seed)
        
        method.fit(X_train, y_train)
        y_pred = method.predict(X_test)

        metric_values.append(accuracy_score(y_test, y_pred))

    kfold_error = np.mean(metric_values)
    
    return kfold_error

# Para análise do melhor alpha para a rede MLP
def grid_search_mlp(X_train, X_test, y_train, y_test, units, epochs):
    grid_search = np.logspace(-2, 0, 11) # Alphas
    val_list = []
    
    for i in range(grid_search.shape[0]):
        alpha = grid_search[i]

        model = models.MLPClassifier(hidden_unit=units, epochs=epochs, alpha=alpha) 
        model.fit(X_train, y_train)

        y_pred = np.argmax(model.predict(X_test))

        wrong_index_val = y_test != y_pred
        val_list.append(np.mean(wrong_index_val))

    best_alpha = grid_search[np.argmin(val_list)] 
    print("[MLP] Melhor modelo encontrado: alpha={}".format(best_alpha))
    
    final_model = models.MLPClassifier(hidden_unit=units, epochs=epochs, alpha=best_alpha) 
    final_model.fit(X_train, y_train)
    
    plot_loss_path(final_model.loss_history(), 'Função de loss ao longo das iterações')
    
    return best_alpha