import tensorflow as tf
import numpy as np
import pandas as pd

class ActivationFunction:

	def relu(x, parameters):
		return x*(x>0)

	def sigmoid(x, parameters):
		return 1/(1+np.exp(-x))

	def tanh(x, parameters):
		return np.tanh(x)

	def softmax(x, parameters):
		x -= np.max(x)
		return  np.exp(x).T / np.sum(np.exp(x), axis=1).T

	def leakyrelu(x, parameters):
		return np.where(x > 0, x, x*0.01)

	def swish(x, parameters = 1):
		return x*sigmoid(parameters*x)

	def binarystep(x, parameters):
		x[x > 0] = 1
		x[x <= 0] = -1
		return x

	def parametricrelu(x, parameters):
		return np.where(x > 0, x, x*parameters)		

	def explinearunit(x, parameters = 1.67):
		x[x <= 0] = parameters*(np.exp(x) - 1)
		return x
		
	def scaledexplinearunit(x, parameters):
		alpha = parameters[0]
		lamda = parameters[1]
		x = explinearunit(x, alpha)
		return lamda*x 

	def softplus(x, parameters):
		if (x > parameters):
    		return x
  		else:
    		return np.log(1.0 + np.exp(x))	
