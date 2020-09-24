import numpy as np

# 1ª questão, a.: Rede MLP para classificação
class MLPClassifier():
    # Coloquei uma taxa de aprendizado (alpha) padrão de 0.1 e 100 épocas
    def __init__(self, hidden_unit=None, epochs=100, alpha=0.1):
        self.units = hidden_unit
        self.epochs = epochs
        self.alpha = alpha
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def grad_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def logistic_loss(self, y, y_pred):
        return np.mean(-y * np.log(y_pred) - (1 - y)*np.log(1 - y_pred))

    # Inicializando os pesos com valores entre 0 a 1
    def initialize_weights(self, X, hidden_unit):
        w_init = np.array([np.random.uniform(low=0, high=1, size=X.shape[1]) for i in range(hidden_unit)]) 
        w_init[:, 0] = 1
        weights = []
        weights.append(w_init)

        w_init = np.array([np.random.uniform(low=0, high=1, size=hidden_unit + 1) for i in range(X.shape[0])]) 
        w_init[:, 0] = 0
        weights.append(w_init)
        
        return weights
    
    # Calcula ui, zi, uk, yk
    def mlp_forward(self, weights, X, activation_function, output=False):
        num_hidden_layers = len(weights) - 1
        layer_u = []
        layer_z = []
        
        for h in range(num_hidden_layers + 1):
            if (h == 0):
                u = weights[h] @ X
                z = np.vstack((np.ones((1, u.shape[1])), activation_function(u)))
            else:
                u = weights[h] @ layer_z[h - 1]
                z = activation_function(u)
            
            layer_u.append(u)
            layer_z.append(z)

        if output: 
            y_pred = layer_z[1]
            return y_pred #Retorna os valores preditos
        else:
            return (layer_u, layer_z)
        
    # Calcula o error/ruído (ek) e os gradientes locais = delta_k e delta_i
    def mlp_backward(self, weights, y, grad_activation_function, layer_u, layer_z):
        num_hidden_layers = len(weights) - 1
        layer_delta = [None]*(num_hidden_layers + 1)
        
        for h in range(num_hidden_layers, -1, -1):
            if h < num_hidden_layers:
                delta = grad_activation_function(layer_u[h]) * \
                            (weights[h + 1][:, 1:].T @ layer_delta[h + 1])
            else:
                y_pred = y - layer_z[1]
                delta = y_pred * grad_activation_function(layer_u[1])
            layer_delta[h] = delta

        return layer_delta

    def fit(self, X, y):    
        activation = self.sigmoid
        grad_activation = self.grad_sigmoid 
        
        loss_function = self.logistic_loss
        loss_history = []

        num_hidden_layers = 1
        
#         y = y.copy()         
#         if len(y.shape) == 1:   
#             y = y[:, None]

        # Inicializando os pesos 
        weights = self.initialize_weights(X, self.units)

        for epoch in range(self.epochs):
            random_permutation = np.array_split(np.random.permutation(y.shape[0]), y.shape[0])
            for i in random_permutation:  
                Xi = X[i].T
                yi = y[i].T

                # Sentido direto
                layer_u, layer_z = self.mlp_forward(weights, Xi, activation)

                # Sentido inverso
                layer_delta = self.mlp_backward(weights, yi, grad_activation, layer_u, layer_z)

                # Atualiza os pesos
                for h in range(num_hidden_layers + 1):    
                    if h == 0:                    
                        layer_input = Xi.T
                    else:
                        layer_input = layer_z[h - 1].T
                    delta_weight = (self.alpha / Xi.shape[1]) * layer_delta[h] @ layer_input
                    weights[h] += delta_weight
            
            model_output = self.mlp_forward(weights, X.T, activation, output=True)
            loss_history.append(loss_function(y.T, model_output[0]))

        self.weights = weights
        self.loss = loss_history

    def predict(self, X):
        model_output = self.mlp_forward(self.weights, X.T, self.sigmoid, output=True)
        
        return np.maximum(0, np.sign(model_output[0] - 0.55)) #Convertendo as probabilidades para 0's e 1's
    
    def loss_history(self):
        return self.loss

# 1. questão, b.: KNN para classificação
class KNeighborsClassifier():
    def __init__(self, n_neighbors=None):
        self.k = n_neighbors
        
    def euclidean_distance(self, Xi, Xj):
        return np.sqrt(np.sum((Xi - Xj)**2))
    
    def batch_euclidean_distance(self, X1, X2):
        return -2 * X1 @ X2.T + np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)
    
    # Fluxo geral e mais específico do algoritmo 
    def distance_matrix(self, X_, X):
        dist_matrix = np.zeros((X_.shape[0], X.shape[0]))
        
        for i in range(X_.shape[0]):
            for j in range(X.shape[0]):
                dist_matrix[i, j] = self.euclidean_distance(X_[i, :], X[j, :])

        return dist_matrix
    
    # Forma mais simplificada, utilizando as operações do próprio Numpy
    def batch_distance_matrix(self, X_, X):
        return self.batch_euclidean_distance(X_, X)

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.X = X
        self.y = y.astype(int)
        
    def predict(self, X_):
        dist_matrix = self.batch_distance_matrix(X_, self.X)
        #dist_matrix = self.distance_matrix(X_, self.X)
        knn = np.argsort(dist_matrix)[:, 0:self.k]
        
        y_pred = self.classes[np.argmax(np.apply_along_axis(np.bincount, 1, self.y[knn], minlength=self.classes.shape[0]), axis=1)]

        return y_pred