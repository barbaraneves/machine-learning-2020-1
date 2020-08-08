import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from decimal import Decimal

from modules import metrics

def plot_line_graphic(X, y, y_pred, X_name, y_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, marker='o', s=10, color='slategray')

    plt.xlabel(X_name)
    plt.ylabel(y_name)

    plt.plot(X, y_pred, color='firebrick');
    
def calculates_error_metrics(models, X_train, y_train, X_test, y_test):
    scores = {}
    scores_tr = {}
    scores_te = {}
    
    pred_values = {}
    pred_values_tr = {}
    pred_values_te = {}
    
    for k, v in models.items():
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

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, train, width, label='Train', color='slategray')
    rects2 = ax.bar(x + width/2, test, width, label='Test', color='firebrick')

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

def modifying_regularization_coef(X_train, y_train, pred_feature, y_true, model, epochs, learning_rate, max_lamb, metric):
    values_metric = []
    
    for l in range(1, max_lamb + 1):
        model.fit(X_train, y_train, epochs, learning_rate, lamb=l)
        y_pred = model.predict(pred_feature)
        values_metric.append(metric(y_true, y_pred))
    
    return values_metric

def plot_lambdas(train_error, test_error):
    plt.figure(figsize=(10, 5))

    x = range(1, 6) 
    plt.plot(x, train_error, marker='o', label="Train", color='slategray')
    plt.plot(x, test_error, marker='o', label = "Test", color='firebrick')

    plt.xlabel('Lambdas')
    plt.ylabel('MSE')

    plt.legend()
    plt.show()