# Módulo que contém todas as funções necessárias para criação e avaliação do PCA
import numpy as np
import matplotlib.pyplot as plt

# Função de covariância
def cov_matrix(m):    
    m -= m.mean(axis=0)  
#     mu = m.mean(axis=0)
#     m = m - mu
    return np.dot(m.T, m.conj()) / (m.shape[0] - 1) 

# Método PCA
class PCA():
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        cm = cov_matrix(X)
        eigen_values, eigen_vectors = np.linalg.eig(cm)
        
        # Proporção dos autovalores sobre o total
        self.proportion_variance_explained = np.cumsum(eigen_values) / np.cumsum(eigen_values)[-1] 
        
        eigen_values = eigen_values[0:self.n_components]
        eigen_vectors = eigen_vectors[:, 0:self.n_components]
        
        # Garantindo que os valores estejam mesmo em ordem decrescente
        descending_order = np.flip(np.argsort(eigen_values))
        eigen_values = eigen_values[descending_order]
        eigen_vectors = eigen_vectors[:, descending_order]

        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors

    def transform(self, X):
        return X @ self.eigen_vectors
    
def plot_variance_explained(prop=[]):
    fig = plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(prop, '-k', color='firebrick')
    
    plt.xlabel("Componentes (dimensões)")
    plt.ylabel("Proporção de variância explicada")
    plt.title("Análise da aplicação do PCA")
    
    plt.show()