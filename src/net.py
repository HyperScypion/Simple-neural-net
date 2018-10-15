import random
import pandas
import numpy as np 

dataset = np.array(([1, 0], [0, 0], [1, 1]), dtype=float)

output = np.array(([0, 0, 1], dtype=float) 

class MLP(object):
    def __init__(self, inputLayer, hiddenLayer, outputLayer):
        
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer

        self.weights0 = np.random.rand(self.inputLayer, self.hiddenLayer)
        self.weights1 = np.random.rand(self.hiddenLayer, self.outputLayer)
    
    def forward(self):
    
        self.z = sigma(np.dot(inputLayer, weights0))
        self.z2 = sigma(np.dot(z2, weights1))
        return self.z2

    def backprop(self):
     

    def sigma(x):
        return (1 / (1 + np.exp(-x))
    
    def sigmaPrime(x):
        return x * (1 - x)
