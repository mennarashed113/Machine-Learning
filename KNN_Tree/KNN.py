import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain


def cal_distance(x1, x2):
    distance=np.sqrt(np.sum((x1-x2)**2))
    return distance


class KNN:

    def __init__(self,k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y).reshape((-1,1))

#0,1,2,3
#[2,5,1,0]
#3,2,0,1
    def _predict(self, x):
        distances = np.array([cal_distance(x, x_train) for x_train in self.X_train]).reshape((-1,1))

        k_indices = np.argsort(distances,axis=0).reshape((-1,1))[:self.k]

        k_nearest_labels = [self.y_train[i[0]][0] for i in k_indices]
        #print(self.y_train)
        #print(k_indices)

        most_common = Counter(k_nearest_labels).most_common()
        if self.k%2==0:
            if most_common[0][1] ==(self.k/2):
                k_distances = [distances[i[0]][0] for i in k_indices]
                weights=[1/(i + 1e-10) for i in k_distances]

                weighted_votes = Counter()
                for label, weight in zip(k_nearest_labels, weights):
                    weighted_votes[label] += weight

                return max(weighted_votes, key=weighted_votes.get)
            else:
                return most_common[0][0]
        else:
            return most_common[0][0]
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
