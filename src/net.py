import random
import numpy as np 

learning_rate = 0.005

def sigma(x):
    return (1 / (1 + np.exp(-x)))

input_X = np.array([ [1, 0], 
		                 [0, 0],
		                 [1, 1] ])

output = np.array([[0, 0, 1]]).T

syn1 = 2*np.random.rand(2, 1) - 1

for i in range(1000): 
    layer = sigma(np.dot(input_X, syn1))
    error = output - layer
    delta = error * sigma(layer)
    syn1 += np.dot(input_X.T, sigma(layer)*(output-layer))

prediction = np.array([[1, 1]])

print(sigma(np.dot(prediction, syn1)))
