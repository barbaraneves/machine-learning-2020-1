import numpy as np

class KNeighborsClassifier():
    def __init__(self, n_neighbors=None):
        self.k = n_neighbors
        
    def euclidean_distance(self, Xi, Xj):
        return np.sum((Xi - Xj)**2)
    
    def batch_euclidean_distance(self, X1, X2):
        return -2 * X1 @ X2.T + np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)
    
    def distance_matrix(self, X_, X):
        dist_matrix = np.zeros((X_.shape[0], X.shape[0]))
        
        for i in range(X_.shape[0]):
            for j in range(X.shape[0]):
                dist_matrix[i, j] = self.euclidean_distance(X_[i, :], X[j, :])

        return dist_matrix
    
    #Explicar o batch
    def batch_distance_matrix(self, X_, X):
        return self.batch_euclidean_distance(X_, X)

    def fit(self, X, y, batch=False):
        self.classes = np.unique(y)
        self.X = X
        self.y = y.astype(int)
        self.batch = batch
        
    def predict(self, X_):
        if self.batch:
            dist_matrix = self.batch_distance_matrix(X_, self.X)
        else:
            dist_matrix = self.distance_matrix(X_, self.X)
        knn = np.argsort(dist_matrix)[:, 0:self.k]
        
        y_pred = self.classes[np.argmax(np.apply_along_axis(np.bincount, 1, self.y[knn], minlength=self.classes.shape[0]), axis=1)]

        return y_pred