import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

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
