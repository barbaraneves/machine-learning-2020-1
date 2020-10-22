# Módulo que contém todas as funções necessárias para criação e avaliação do DecisionTreeClassifier
import numpy as np

class Node:
    def __init__(self, y_pred):
        self.y_pred = y_pred
        self.index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier():
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
    
    # Conta a frequência de cada classe para um nó 
    def classes_frequencies(self, y):
        return [sum(y == (c + 1)) for c in range(self.num_classes)]
    
    # Encontra o melhor split para um nó da árvore
    def best_split(self, X, y):
        n = X.shape[0]

        num_parent = self.classes_frequencies(y)
        best_gini = 1.0 - sum((m / n) ** 2 for m in num_parent) # Valor do Gini para o nó atual
        best_idx, best_thr = None, None
        
        for idx in range(self.num_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0]*self.num_classes
            num_right = num_parent.copy()
            
            for i in range(1, n):
                c = int(classes[i - 1]) - 1 # As classes são 1, 2 e 3 
                num_left[c] += 1
                num_right[c] -= 1
                
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.num_classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (n - i)) ** 2 for x in range(self.num_classes)
                )
                
                # A impureza de um split (pai) se dá pela média ponderada da impureza dos filhos
                gini = (i * gini_left + (n - i) * gini_right) / n 
                
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx  # Índice da feature para o melhor slip, ou None, para nenhum split
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2 # Limite usado no split, ou None, para nenhum split
        return best_idx, best_thr
        
    def grow_tree(self, X, y, depth=0):
        num_samples_per_class = self.classes_frequencies(y)
        y_pred = np.argmax(num_samples_per_class)
        node = Node(y_pred=y_pred)
        
        # Dividindo o dataset recursivamente até atingir o max_depth 
        # Condição baseada na implementação do Sklearn
        if depth < self.max_depth:
            idx, thr = self.best_split(X, y)
            
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
               
                node.index = idx
                node.threshold = thr
                node.left = self.grow_tree(X_left, y_left, depth + 1)
                node.right = self.grow_tree(X_right, y_right, depth + 1)
        return node
    
    def fit(self, X, y):
        self.num_classes = len(set(y)) 
        self.num_features = X.shape[1]
        self.tree_ = self.grow_tree(X, y)
        
    def _predict(self, Xi):
        node = self.tree_
        
        while node.left:
            if Xi[node.index] < node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.y_pred + 1 # As classes do dataset deste trabalho são 1, 2 e 3 

    def predict(self, X):
        return [self._predict(i) for i in X]