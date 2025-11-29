import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import Counter

import numpy as np
import heapq as hq
import math
import matplotlib.pyplot as plt
import time

class KNN:
    def __init__(self, k, encoder_type, dist_met):
        self.encoder_type = encoder_type
        self.k = k
        self.dist_name = dist_met
        self.distance_metric = self._get_distance_metric()
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((np.square(x1-x2))))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _cosine_distance(self, x1, x2):
        return 1-(np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2)))
    
    def _get_distance_metric(self):
        if self.dist_name == 'euclidean':
            return self._euclidean_distance
        elif self.dist_name == 'manhattan':
            return self._manhattan_distance
        elif self.dist_name == 'cosine':
            return self._cosine_distance
        else:
            raise ValueError("Unsupported encoder type")
            
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # This function below is inspired from ChatGPT
    def predict(self, X_test):
        y_pred = []
        k_nearest = []
        for x in X_test:
            distance_metric = self._get_distance_metric()
            distances = [self.distance_metric(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            y_pred.append(most_common[0][0])
        return y_pred
    
    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=1, average='weighted')
        recall = recall_score(y_val, y_pred, zero_division=1, average='weighted')
        return f1, accuracy, precision, recall
    
    # Another function to verify outputs
    def evaluate2(self, X_val, y_val):
        y_pred = self.predict(X_val)
        correct = np.sum(np.array(y_pred) == np.array(y_val))
        accuracy = correct / len(y_val)
        
        tp = np.sum(np.logical_and(np.array(y_pred) == 1, np.array(y_val) == 1))
        fp = np.sum(np.logical_and(np.array(y_pred) == 1, np.array(y_val) == 0))
        fn = np.sum(np.logical_and(np.array(y_pred) == 0, np.array(y_val) == 1))

        if (tp+fp) == 0: 
            precision = 1 
        else:
            precision = tp / (tp + fp)

        if (tp+fn) == 0: 
            recall = 1 
        else:
            recall = tp / (tp + fn)
            
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return accuracy, precision, recall, f1
    

train_data = np.load('data.npy', allow_pickle=True)
X_train = [row[2][0] for row in train_data]
y_train = [row[3][0] for row in train_data] 

test_data_file = sys.argv[1]
test_data = np.load(test_data_file, allow_pickle=True)
X_test = [row[2][0] for row in test_data]
y_test = [row[3][0] for row in test_data]

knn = KNN(k = 13, encoder_type = 'VIT', dist_met = 'manhattan')
knn.fit(X_train, y_train)
f1, accuracy, precision, recall = knn.evaluate(X_test, y_test)

# print scores on terminal
print('Accuracy = {}, Precision = {}, Recall = {}, F1 Score = {}\n'.format \
                  (accuracy, precision, recall, f1))
