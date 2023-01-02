import numpy as np
import pandas as pd

import networkx as nx
from networkx.algorithms import node_classification

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


class HarmonicFunctionSSL():
    def __init__(self, X_train, y_train, k=10) -> None:
        self.X_train = X_train
        self.y_train = y_train

        self.k = k

    def generate_cosine_weight_matrix(self, X):
        return cosine_similarity(X)

    def KNN_sparify(self, W, y):
        k = self.k
        dim = W.shape
        P = np.zeros(dim)
        P_ = np.zeros(dim)
        labeled = (y[y != -1])
        V = np.zeros((dim[0], len(np.unique(labeled))))
        

        label_idx = (y != -1).nonzero()[0]
        P_[:, label_idx] = 1
        P_[label_idx, :] = 1
        
        
        
        for i in label_idx:
            # print(i, y[i])
            V[i, int(y[i])] = 1
        
        
        most_match_k_adj = np.argpartition(W, -k, axis=1)[:, -k:]
        most_match_labeled = np.argmax(W*P_, axis=1)
        
        print(len(most_match_labeled))
        
        for i, _ in enumerate(W):
            P[i, most_match_k_adj[i]] = 1
            P[i, most_match_labeled[i]] = 1
            
            
        return P, V
    def create_weight_matrix(self, X, y):
        W = self.generate_cosine_weight_matrix(X)
        P, V = self.KNN_sparify(W, y)
        
        
        W = W*P
        np.fill_diagonal(W, 0)
        
        return W

    def construct_graph(self, W, y):
        G = nx.from_numpy_matrix(W)
        
        for idx, _ in enumerate(y):
            if y[idx] != -1:
                G.nodes[idx]['label'] = y[idx]
        
        return G

    def inference(self, X_test, y_test):
        labeled_idx = len(self.X_train)
        un_y_test = np.ones(len(y_test))*-1
        X = np.append(self.X_train, X_test, axis=0)
        y = np.append(self.y_train, un_y_test, axis=0)
        
        W = self.create_weight_matrix(X, y)
        G = self.construct_graph(W, y)
        
        pred = node_classification.harmonic_function(G)
        
        pred_test = pred[labeled_idx:]
        
        
        print("Accuracy score = ", accuracy_score(y_test, pred_test))
    
        return pred_test

    