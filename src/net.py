import random
import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([0, 0], [1, 0], [0, 1]), dtype=float)
y = np.array(([1], [0], [0]), dtype=float)

class MultiLayerPerceptron():
  def __init__(self, inputSize, hiddenSize, outputSize):
    
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.hiddenSize = hiddenSize

    
    self.weight0 = np.random.rand(self.inputSize, self.hiddenSize) 
    print("Weight0: " + str(self.weight0))
    self.weight1 = np.random.rand(self.hiddenSize, self.outputSize)
    print("Weight1: " + str(self.weight1))

  def forward(self, X):
    
    self.hidden = self.sigma(np.dot(X, self.weight0))  
    o = self.sigma(np.dot(self.hidden, self.weight1)) 
    return o 

  def sigma(self, x):
     
    return 1/(1+np.exp(-x))

  def sigmaPrime(self, x):
    
    return x * (1 - x)

  def backward(self, inputX, output, o):
   
    self.o_error = output - o
    self.o_delta = self.o_error*self.sigmaPrime(o) 
    self.z2_error = self.o_delta.dot(self.weight1.T) 
    self.z2_delta = self.z2_error*self.sigmaPrime(self.hidden)

    self.weight0 += inputX.T.dot(self.z2_delta) 
    self.weight1 += self.hidden.T.dot(self.o_delta) 

  def train (self, inputX, output):
    o = self.forward(inputX)
    self.backward(inputX, output, o)

nn = MultiLayerPerceptron(2, 2, 1)
for i in range(2000): 
  nn.train(X, y)

print(nn.forward([0, 0]))
print(nn.forward([1, 1]))