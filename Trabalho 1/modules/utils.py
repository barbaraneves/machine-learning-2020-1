import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from decimal import Decimal

from modules import metrics

def plot_line_graphic(X, y, y_pred, X_name, y_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, marker='o', s=10)

    plt.xlabel(X_name)
    plt.ylabel(y_name)

    plt.plot(X, y_pred, color="g");
    
def calculates_error_metrics(clfs, X_train, y_train, X_test, y_test):
    scores = {}
    scores_tr = {}
    scores_te = {}
    
    pred_values = {}
    pred_values_tr = {}
    pred_values_te = {}
    
    for k, v in clfs.items():
        y_pred_train = v.predict(X_train)
        y_pred_test = v.predict(X_test)
        
        pred_values_tr[k] = y_pred_train
        pred_values_te[k] = y_pred_test
            
        scores_tr[k + '_MSE'] = metrics.MSE(y_train, y_pred_train)
        scores_tr[k + '_R2'] = metrics.R2(y_train, y_pred_train)

        scores_te[k + '_MSE'] = metrics.MSE(y_test, y_pred_test)
        scores_te[k + '_R2'] = metrics.R2(y_test, y_pred_test)
        
    pred_values['Train'] = pred_values_tr
    pred_values['Test'] = pred_values_te
    
    scores['Train'] = scores_tr
    scores['Test'] = scores_te

    return pred_values, scores

def plot_scores_histogram(linear_models, scores, score_name):
    
    labels = [model + '_' + score_name for model in list(linear_models.keys())]
    train = [float(new_l.quantize(Decimal('.001'), rounding="ROUND_DOWN")) for new_l in [Decimal(scores['Train'][l]) for l in labels]]
    test = [float(new_l.quantize(Decimal('.001'), rounding="ROUND_DOWN")) for new_l in [Decimal(scores['Test'][l]) for l in labels]]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width/2, train, width, label='Train')
    rects2 = ax.bar(x + width/2, test, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title('Scores by set', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.legend(fontsize=14)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    plt.show()