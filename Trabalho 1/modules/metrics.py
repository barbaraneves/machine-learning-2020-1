import numpy as np

# RSS será usada para calcular as métricas MSE e R2
def RSS(y_true, y_pred):
    rss = np.sum((y_pred - y_true)**2)
    
    return rss

# a. MSE
def MSE(y_true, y_pred):
    n = len(y_true)
    
    rss = RSS(y_true, y_pred)
    mse = rss/n
    
    return mse

# b. R2
def R2(y_true, y_pred):
    rss = RSS(y_true, y_pred)
    tss = np.sum((y_true - np.mean(y_true))**2)

    r2 = 1 - (rss/tss)
    
    return r2