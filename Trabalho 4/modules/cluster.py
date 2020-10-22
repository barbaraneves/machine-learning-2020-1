# Módulo que contém todas as funções necessárias para criação e avaliação do K-Means
import numpy as np
import matplotlib.pyplot as plt

# Método K-Means
class KMeans():
    def __init__(self, n_clusters=2, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def euclidean_distance(self, Xi, Xj):
        return np.sqrt(np.sum((Xi - Xj)**2))
    
    #Inicializa os centróides iniciais de maneira arbitrária 
    def initial_centers(self, X, k): 
        centroids = []
        n = X.shape[1]
        
        min_ = np.min(X, axis=0)
        max_ = np.max(X, axis=0)
        
        for i in range(k):
            centroids.append(np.random.uniform(min_, max_, n))
        return np.array(centroids)
    
    #Calcula o índice do centroid mais próximo para cada ponto do dataset
    def nearest_centroids(self, X, cluster_centers):
        nearest_indexes = []
        
        for i in range(X.shape[0]):
            dist = [self.euclidean_distance(X[i], center) for center in cluster_centers]
            nearest_index = [index for index, val in enumerate(dist) if val==min(dist)]
            nearest_indexes.append(nearest_index[0])
        
        return nearest_indexes
    
    #Soma das distâncias quadradas das amostras para o centro do cluster mais próximo
    def inertia(self, X, cluster_centers, nearest_indexes): 
        return np.sum([self.euclidean_distance(X[i], cluster_centers[nearest_indexes[i]])**2 for i in range(0, len(X))])
    
    # Atualiza os centroids
    def update_centroids(self, X, nearest_indexes):
        D = max(np.unique(nearest_indexes)) + 1 # Dimensão
        sum_ = np.zeros((D, X.shape[1]))
        total = np.zeros(D)
        
        for i in range(0, len(X)):
            sum_[nearest_indexes[i]] += X[i]
            total[nearest_indexes[i]] += 1

        cluster_centers = [np.divide(sum_[i], total[i], where=sum_[i] != 0) for i in range(0, D)] 
        return np.array(cluster_centers)
    
    # Calcula os centros dos clusters e prediz o índice dos clusters para cada amostra
    def fit_predict(self, X):
        # Inicializa os centróides
        cluster_centers = self.initial_centers(X, self.n_clusters)
        
        # Computa o cluster de cada amostra
        cluster_indexes = self.nearest_centroids(X, cluster_centers)

        # Calcula a inércia inicial
        init_inertia = self.inertia(X, cluster_centers, cluster_indexes)
        
        for i in range(0, self.max_iter):
            cluster_centers = self.update_centroids(X, cluster_indexes)
            cluster_indexes = self.nearest_centroids(X, cluster_centers)
            inertia_ = self.inertia(X, cluster_centers, cluster_indexes)
            if(init_inertia == inertia_):
                break
            else:
                init_inertia = inertia_  
        
#         self.model = {'cluster_centers': cluster_centers, 'labels': cluster_indexes, 'inertia': inertia}
        
        self.cluster_centers_ = cluster_centers
        self.labels_ = cluster_indexes
        self.inertia_ = inertia_
        
#     def fit_predict(self, X):
#        self.fit(X)
#        return self.model

def kmeans_elbow_visualizer(X, k_range=[2], max_iter=50):
    inertia_list = []

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16,12))

    for i, K in enumerate(k_range):
        kmeans = KMeans(n_clusters=K, max_iter=max_iter)
        kmeans.fit_predict(X)

        labels = np.unique(kmeans.labels_)
        y_kmeans = kmeans.labels_
        inertia_list.append(kmeans.inertia_)

        colors = plt.cm.Set1(np.linspace(0, 0.9, labels.shape[0]))

        axs.flat[i].set_title("K = {}; Inércia = {}".format(K, kmeans.inertia_), fontsize=14)

        for j, label in enumerate(labels):
            axs.flat[i].scatter(X[y_kmeans==label, 0], X[y_kmeans==label, 1], s=100, marker='o', color=colors[j]) 

    #     for k in range(K):
        axs.flat[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, facecolors='w', edgecolors='k', linewidth=3)
        axs.flat[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=20, color='k')

    #     axs.flat[i].set_xlim(xlim)
    #     axs.flat[i].set_ylim(ylim)

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 14})

    plt.plot(k_range, inertia_list, '-k', color='firebrick')
    plt.xlabel('Número de clusters (k)', fontsize=14)
    plt.ylabel('Distâncias', fontsize=14)

    plt.show()