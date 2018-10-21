import random
import pandas
import numpy as np 

dataset = np.array(([1, 0], [0, 0], [1, 1]), dtype=float)

output = np.array(([0, 0, 1], dtype=float) 
def sigma(x):
	return (1 / 1 + np.exp(-x))
		
def SigmaPrime(x):
	return x + (1 - x)
		
class MultiLayerPerceptron():
	
	def __init__(self, inputLayer, hiddenLayer, outputLayer):
		self.inputLayer = inputLayer
		self.hiddenLayer = hiddenLayer
		self.outputLayer = outputLayer
	
		self.weight0 = np.random.rand(self.inputLayer, self.hiddenLayer) 
		self.weight1 = np.random.rand(self.hiddenLayer, self.outputLayer)
	
		print("Pierwsza macierz wag")
		print(self.weight0)
		print("Druga macierz wag")
		print(self.weight1)
	
	def forward(self, input):
		
		self.hidden = sigma(np.dot(input, self.weight0))
		print("First layer")
		print(self.hidden)
		self.out = sigma(np.dot(self.hidden, self.weight1))
		print("Output")
		print(self.out)
		
		
	def backprop(self, input, output):
		print("backprop")
		# weight1B = np.dot(self.hidden, (2*(output - self.out) * SigmaPrime(self.out)))
		# weight0B = np.dot(input.T, (np.dot(2*(output - self.out) * SigmaPrime(self.out), self.weight1.T) * SigmaPrime(self.hidden)))
		# self.weight0 += weight0B
		# self.weight1 += weight1B
	

	def train(self, input, output):
		print("train")
		
net = MultiLayerPerceptron(3, 4, 2)
dataset = np.array([1, 2, 3])
net.forward(dataset)
output = np.array([1, 2])
# net.backprop(dataset, output)
