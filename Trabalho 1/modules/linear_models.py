import numpy as np
import pandas as pd

from modules import metrics

# a. Regressão Linear Univariada - Método Analítico
class LRAnalyticalMethod():
    def __init__(self):
        pass

    def fit(self, X, y):
        # Número de observações
        n = len(X)
  
        # Média do X e do y
        mean_x, mean_y = np.mean(X), np.mean(y) 
        
        # Aplicando método analítico
        somat_xy = (np.sum(y * X)) - (n * mean_y * mean_x)
        somat_xx = (np.sum(X * X)) - (n * mean_x * mean_x) 
 
        b_1 = somat_xy / somat_xx 
        b_0 = mean_y - (b_1 * mean_x) 
        
        self.b_1 = b_1
        self.b_0 = b_0

    def predict(self, X):
        return self.b_0 + self.b_1 * X
    
    def coef_(self):
        return [self.b_0, self.b_1]
    
# b. Regressão Linear Univariada - Gradiente Descendente
class LRGradientDescent():
    def __init__(self): 
        pass 

    def fit(self, X, y, epochs, learning_rate): 
        # Inicializando os coeficientes com 0
        b_0 = 0
        b_1 = 0
        
        n = len(X)  # Número de observações
        cost = np.zeros(epochs)
        
        # Aplicando gradiente descendente
        for e in range(epochs):
            y_pred = b_0 + b_1 * X
      
            # Calculando derivadas (gradientes)
            D_0 = (1/n) * sum(y - y_pred)
            D_1 = (1/n) * sum((y - y_pred) * X) 
            
            # Atualizando betas
            b_0 = b_0 + learning_rate * D_0  
            b_1 = b_1 + learning_rate * D_1
            
            cost[e] = metrics.MSE(y, y_pred)
            
        self.b_0 = b_0
        self.b_1 = b_1
        self.cost = cost
  
    def predict(self, X):  
        return self.b_0 + self.b_1 * X
    
    def coef_(self):
        return [self.b_0, self.b_1]
    
    def cost_history(self): # Apenas para escolher uma época boa
        return self.cost
    
# c. Regressão Linear Multivariada - Método Analítico
class MLRAnalyticalMethod():
    def __init__(self):
        pass

    def fit(self, X, y):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]

        beta = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
        
        self.B = beta
        
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        
        return X_ @ self.B
    
    def coef_(self):
        return self.B
    
# d. Regressão Linear Multivariada - Gradiente Descendente
class MLRGradientDescent():
    def __init__(self):
        pass

    def fit(self, X, y, epochs, learning_rate):
        n = X.shape[0] # Número de amostras
        p = X.shape[1] # Número de variáveis (parâmetros)
        X_ = np.c_[np.ones(n), X]
        
        cost = np.zeros(epochs)
        B = np.zeros(p + 1)
         
        for e in range(epochs):    
            y_pred = X_.dot(B)
            
            D = (1/n) * (X_.T.dot(y_pred - y))
            B = B - learning_rate * D
            cost[e] = metrics.MSE(y, y_pred)
            
        self.B = B
        self.cost = cost
        
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        
        return X_.dot(self.B)
    
    def coef_(self):
        return self.B
    
    def cost_history(self):
        return self.cost
    
# e. Regressão Linear Multivariada - Gradiente Descendente Estocástico
class MLRStochasticGradientDescent():
    def __init__(self):
        pass

    def fit(self, X, y, epochs, learning_rate):
        n = X.shape[0] # Números de amostras
        p = X.shape[1] # Número de variáveis (atributos)
        X_ = np.c_[np.ones(n), X]
        
        cost = np.zeros(epochs)
        B = np.zeros(p + 1)
         
        for e in range(epochs): 
            count = 0.0
            random_permutation = np.random.permutation(n)
            for Xi, yi in zip(X_[random_permutation], y[random_permutation]):
                y_pred = Xi.dot(B)

                B = B - learning_rate * (Xi.T.dot(y_pred - yi))
                count += metrics.MSE(y, y_pred)
            cost[e] = count
            
        self.B = B
        self.cost = cost
        
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        
        return X_.dot(self.B)
    
    def coef_(self):
        return self.B
    
    def cost_history(self):
        return self.cost

# f. Regressão Quadrática usando Regressão Múltipla
class QuadraticLN():
    def __init__(self): 
        pass 

    def fit(self, X, y): 
        X_ = X**2
            
        self.MLR = MLRAnalyticalMethod()
        self.MLR.fit(X_, y)

    def predict(self, X):
        X_ = X**2 
        return self.MLR.predict(X_)
    
    def coef_(self):
        return self.MLR.coef_()
    
# g. Regressão Cúbica usando Regressão Múltipla
class CubicLN():
    def __init__(self): 
        pass 

    def fit(self, X, y): 
        X_ = X**3
            
        self.MLR = MLRAnalyticalMethod()
        self.MLR.fit(X_, y)

    def predict(self, X):
        X_ = X**3
        return self.MLR.predict(X_)
    
    def coef_(self):
        return self.MLR.coef_()

# h. Regressão Linear Regularizada Multivariada - Gradiente Descendente
class MLRRegularized():
    def __init__(self):
        pass

    def fit(self, X, y, epochs, learning_rate, lamb):
        n = X.shape[0] # Números de amostras
        p = X.shape[1] # Número de variáveis (atributos)
        X_ = np.c_[np.ones(n), X]
        
        cost = np.zeros(epochs)
        B = np.zeros(p + 1)
        b_0 = 0
         
        for e in range(epochs): 
            y_pred = X_.dot(B)

            D_0 = (1/n) * sum(y - y_pred) 
            b_0 = b_0 + learning_rate * D_0

            D = ((1/n) * X_.T.dot(y_pred - y)) - (lamb/n * B)
            B = B - learning_rate * D
            cost[e] = metrics.MSE(y, y_pred)
        B[0] = b_0
        
        self.B = B
        self.cost = cost
        
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        
        return X_.dot(self.B)
    
    def coef_(self):        
        return self.B
    
    def cost_history(self):
        return self.cost