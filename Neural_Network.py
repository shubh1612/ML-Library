import tensorflow as tf
import numpy as np
import pandas as pd
import ActivationFunction as af

class NeuralNetwork:
	
	def __init__(self, num_hidden_layers = 1, hidden_layer_sizes = [], hidden_act_func = [], output_act_func = 'softmax', loss = 'cross-entropy'):
		
		self.HiddenLayers = num_hidden_layers
		
		if (len(hidden_act_func) == 0):
			hidden_act_func = ['Relu']*num_hidden_layers

		elif (len(hidden_act_func) != num_hidden_layers):
			print('Either input activation function for all hidden layers or use default')
			exit()

		if (len(hidden_layer_sizes) == 0):
			hidden_layer_sizes = [1]*num_hidden_layers

		elif (len(hidden_layer_sizes) != num_hidden_layers):
			print('Either input size for all hidden layers or use default')
			exit()

		self.HiddenActivation = hidden_act_func
		self.HiddenSizes = hidden_layer_sizes
		self.OutputActivation = output_act_func
		self.Loss = loss
		self.X = None
		self.Y = None

	def initialize_parameters(self, n_x, n_y):

		np.random.seed(2)

		W = []
		b = []
		prev = n_x

		for i in range(self.HiddenLayers):
			curr = self.HiddenSizes[i]
			W.append(np.random.randn(curr, prev)*0.01)
			b.append(np.random.randn((curr, 1)))
			prev = curr

		curr = n_y

		W.append(np.random.randn(curr, prev)*0.01)
		b.append(np.random.randn((curr, 1)))

		self.W = W
		self.b = b

	def ActivationValue(ActivationFunction, X, parameters):
		if(ActivationFunction == 'ReLu'):
			return af.relu(X, parameters)
		elif(ActivationFunction == 'Sigmoid'):
			return af.sigmoid(X, parameters)
		elif(ActivationFunction == 'Tanh'):
			return af.tanh(X, parameters)
		elif(ActivationFunction == 'Softmax'):
			return af.softmax(X, parameters)
		elif(ActivationFunction == 'LeakyRelu'):
			return af.leakyrelu(X, parameters)
		elif(ActivationFunction == 'Swish'):
			return af.swish(X, parameters)
		elif(ActivationFunction == 'BinaryStep'):
			return af.binarystep(X, parameters)
		elif(ActivationFunction == 'ParametricRelu'):
			return af.parametricrelu(X, parameters)
		elif(ActivationFunction == 'ExpLinearUnit'):
			return af.explinearunit(X, parameters)
		elif(ActivationFunction == 'ScaledExpLinearUnit'):
			return af.scaledexplinearunit(X, parameters)
		elif(ActivationFunction == 'SoftPlus'):
			return af.softplus(X, parameters)

		
	def forward_propagation(self, X):

		prev = X

		for i in range(self.HiddenLayers):

			w = self.W[i]
			B = self.b[i]
			z = np.dot(w, prev) + B
			a = ActivationValue(self.HiddenActivation[i], z)

			prev = a
			self.Z.append(z)
			self.A.append(a)

		w = self.W[self.HiddenLayers]
		B = self.b[self.HiddenLayers]
		z = np.dot(w, prev) + B
		a = ActivationValue(self.OutputActivation, z)
		self.Z.append(z)
		self.A.append(a)

	def compute_cost(self, parameters):
	def backward_propagation(self, parameters):

	def backward_propagation(self, parameters, cache, X, Y)
