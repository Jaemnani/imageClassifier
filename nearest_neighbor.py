import numpy as np
from tqdm import tqdm

class NearestNeighbor:
    def __init__(self, k=3, type="euclidean"):
        self.x_train = None
        self.y_train = None
        self.k = 3
        self.type = "euclidean"
        
        self.y_pred = None
        print("Nearest Neighbor Init")
    
    def train(self, x, y):
        self.x_train = x
        self.y_train = y
        
    def predict(self, x):
        num_test = len(x)
        
        y_pred = np.zeros(num_test, dtype = self.y_train.dtype)
        
        for i in tqdm(range(num_test)):
            
            distances = self.calc_distance_all(self.x_train, x[i], type=self.type)

            nearest_neighbors = np.argsort(distances)[:self.k]
            
            y_pred[i] = np.argmax(np.bincount(self.y_train[nearest_neighbors]))
        
        self.y_pred = y_pred
        
        return y_pred
    
    def calc_distance_all(self, x_train, x_test, type="euclidean"):
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.flatten()
        x_diff = x_train - x_test
        if type.lower() == "euclidean":
            return np.sqrt(np.sum(x_diff**2, axis=1))
        elif type.lower() == "manhattan":
            return np.sum(np.abs(x_diff), axis=1)
        else:
            print('type is wrong. \n1. manhattan\n2. euclidean ')
            return -1
        
    def print_result(self, test_labels):
        if self.y_pred != None:
            result_percentage = np.sum(test_labels == self.y_pred) / 100.
            print("Correct Ratio : ", result_percentage)
        else:
            print("Predict First.")


            
import mnist
dataset = mnist.DATASET_MNIST()
classifier = NearestNeighbor()
classifier.train(dataset.train_images, dataset.train_labels)
pred = classifier.predict(dataset.test_images)
classifier.print_result(dataset.test_labels)

import cifar10
dataset = cifar10.DATASET_CIFAR10()
classifier = NearestNeighbor()
classifier.train(dataset.train_images, dataset.train_labels)
pred = classifier.predict(dataset.test_images)
classifier.print_result(dataset.test_labels)

    